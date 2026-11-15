# Deployment and operations (FootyMind 1.0)

FootyMind ships as one container: the FastAPI backend serves the API under `/api` and the built
web lab at `/`. The SkillCorner-only public demo is deployed on Railway; this page also documents how to
run it independently. Data licences and the licence-clear public-demo configuration are in
[`DATA_LICENSES.md`](DATA_LICENSES.md).

## Build

```bash
docker build -t footymind .
```

Two stages: `node:22-slim` builds `web/dist`; `python:3.12-slim` installs the locked runtime
dependencies (`uv sync --frozen --no-dev`) and copies the package, the built lab and the
committed `experiments/results` (validation summaries the API serves). **No match data is in
the image.** Base images are overridable (`--build-arg NODE_IMAGE=... PYTHON_IMAGE=...`).
The image defaults to an unprivileged user (uid 10001). For TLS-intercepting proxies,
configure the custom CA in the base images or build environment; the Dockerfile does
not use BuildKit secret mounts.

## Run

```bash
docker run -d --name footymind -p 8000:8000 -v footymind-data:/data footymind
```

| Variable | Default (image) | Meaning |
|---|---|---|
| `FOOTYMIND_SOURCES` | `metrica-sample-1 metrica-sample-2 metrica-sample-3` | sources to download and ingest: match ids or the groups `metrica`, `skillcorner`, `all` |
| `FOOTYMIND_DATA_DIR` | `/data` (the volume) | raw downloads and processed bundles |
| `FOOTYMIND_WEB_DIST` | `/app/web/dist` | built lab served at `/` |
| `FOOTYMIND_EXPERIMENTS_DIR` | `/app/experiments` | committed results read for validation summaries |
| `FOOTYMIND_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | comma-separated origins allowed to call the API from another origin (only needed when the lab is served elsewhere; same-origin use needs none) |
| `FOOTYMIND_RATE_LIMIT_PER_MIN` | `120` | per-client `/api` requests per minute (burst of the same size); over it, 429 with `Retry-After`; `0` disables. Loopback clients and `/api/health` are exempt |
| `FOOTYMIND_CLIENT_IP_HEADER` | unset | header a reverse proxy fills with the client address (e.g. `X-Real-IP`); unset, every request is keyed by its peer address, so behind a proxy all clients share one bucket (a global cap) |
| `FOOTYMIND_MAX_HEAVY_REQUESTS` | `4` | scenarios, frame analyses, Defensive Shadow and similar-moments requests running at once; others wait |
| `FOOTYMIND_HEAVY_WAIT_S` | `15` | how long a heavy request waits for a slot before a 503 with `Retry-After` |

Sizing, measured on the Metrica default: about 180 MB of raw data and 70 MB of bundles in the
volume, first start about a minute (download, ingest, similar-moments index). Adding all 20
SkillCorner matches adds about 1.8 GB and 120 MB, and about 2.5 minutes of indexing on 4 vCPU.
Warm latency is in C67 (frame analysis about 70 ms p50; a scenario with the 24-member ensemble
about 0.5 s).

**Health.** `GET /api/health` answers 200 with `matches` (servable) and `unavailable` (bundles
whose manifest is from another pipeline version or unreadable; they need `footymind ingest`).
A source that was never ingested appears in neither list. A damaged data file under a valid
manifest stays listed and answers 503 with the fix when opened; the similar-moments index is
built from the readable bundles only. The image's `HEALTHCHECK` polls it with a 300 s start period,
long enough for a first start that downloads and indexes.

**Limits.** Request bodies over 1 MB get 413; every query and body parameter is bounded and
validated (422 without echoing the input); unknown ids are 404; missing or stale derived data is
503 with the command that fixes it; no response contains NaN or Infinity (C68). Each client is
rate-limited and heavy computations are capped (variables above); responses carry
`X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` and `X-Frame-Options: DENY`.
The API has no authentication (it serves public open data read-only). These in-process limits
protect a small instance; a service with real traffic should still add edge rate limiting.

**Privileges.** The image runs as the unprivileged user `app` (uid 10001). If a platform starts
it as root (for example to write a root-owned volume), the entrypoint gives the data directory
to `app` and drops root before doing anything else; the server never runs as root. If that is
impossible it logs `WARNING: running as root` and continues. Both paths are tested in CI
(`scripts/container_recovery.py`, scenario R).

## Start-up, upgrades and recovery

The entrypoint (`docker/entrypoint.sh`) makes the volume servable before starting the server:

* **Empty volume:** download (SHA-256 verified) and ingest `FOOTYMIND_SOURCES`, then build the
  similar-moments index. A failure here stops the container (nothing to serve).
* **Existing volume:** re-ingest bundles written by another pipeline version (an image
  upgrade), ingest any source in `FOOTYMIND_SOURCES` that has no bundle (a first start that
  failed part-way), one source at a time, and rebuild a missing or stale index (also when a
  match was added). A source that fails is logged and is not served; the server still starts.

These paths are tested in the built image in CI (`scripts/container_recovery.py`, four
scenarios; see [`VERIFICATION.md`](VERIFICATION.md)).

**Upgrade:** pull or build the new image and start it on the same volume. Bundles from an older
pipeline version are re-ingested from the raw files already in the volume.

**Rollback:** start the previous image on the same volume. If the newer image had bumped the
pipeline version, an older image from 1.0 onward re-ingests the bundles at its own version on
start (the raw files are kept), so no manual step is needed. For an image from before 1.0, use a
fresh volume.

**Repair a single match** (for example a damaged data file, which answers 503 naming the fix):

```bash
docker exec footymind footymind ingest metrica-sample-2
docker exec footymind footymind build-similarity-index --provider metrica
```

A running server picks up a re-ingest or an index rebuild on the next request; no restart.

**Reset:** remove the volume (`docker volume rm footymind-data`) and start again.

## Without Docker

```bash
uv sync --frozen --no-dev
uv run footymind download metrica && uv run footymind ingest metrica
uv run footymind build-similarity-index --provider metrica
(cd web && npm ci && npm run build)
uv run footymind serve --host 0.0.0.0 --port 8000
```

`footymind serve` runs a single uvicorn worker; caches are per process.

## Railway public demo (free-trial configuration)

The [public SkillCorner demo](https://footymind-web-production.up.railway.app/) runs this
Dockerfile in one Railway service. Its data lives in a separately attached persistent volume
mounted at `/data`. Railway rejects Dockerfile `VOLUME` instructions and BuildKit
`RUN --mount=type=secret` directives.

- Source: `co-rtex/footymind`, branch `main`, Dockerfile at repository root.
- HTTP port: `8000`; health check: `/api/health`.
- Volume: `/data`, 500 MB.
- `FOOTYMIND_SOURCES="skillcorner-1874553 skillcorner-1899585 skillcorner-2017461"`.
- `FOOTYMIND_DATA_DIR=/data` and `PORT=8000`.
- `RAILWAY_RUN_UID=0`: Railway's volume is root-owned and not writable by uid 10001, so
  the container starts as root; the entrypoint then hands `/data` to `app` and the server
  runs as uid 10001 (the deploy log shows "dropping privileges").
- `FOOTYMIND_CLIENT_IP_HEADER=X-Real-IP`, so the rate limit is keyed per visitor rather than
  shared by every request arriving from Railway's edge proxy. The server keys on the header's
  last value, so a client-sent copy cannot pick a fresh bucket when the proxy appends its own.
- `FOOTYMIND_RATE_LIMIT_PER_MIN=60` and `FOOTYMIND_MAX_HEAVY_REQUESTS=2`, tighter than the
  defaults (120 and 4) for the 0.5 vCPU instance.
- Single replica capped at 0.5 vCPU and 0.5 GB RAM; restart on failure
  (two retries).

The public demo excludes Metrica files because their redistribution terms are
not sufficiently clear for public demo hosting. The API has no authentication; its in-process
rate limit and concurrency cap bound what one client can make the 0.5 vCPU instance do, but it
should not be marketed as hardened production infrastructure without edge request limits.
Railway free-trial credits and resource limits can cause the demo to become unavailable when
exhausted.

**Similar moments on the demo.** The demo's index holds 3 matches, while the held-out
validation of the k = 50 estimate (C36) used all 20 SkillCorner matches, so the lab says
"not validated here" and states both counts instead of "validated held-out".
