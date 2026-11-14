# Release verification: container runtime, clean setup, API robustness

_2026-11-14. Numbers quoted here come from committed files or from tests; latency figures are
in `experiments/results/api_latency_v1/results.json`._

## How to re-run

```bash
scripts/container_smoke.sh             # build the image, run it, check API + web + Playwright flow
scripts/container_smoke.sh --no-build  # reuse footymind:smoke
python3 scripts/smoke_api.py http://127.0.0.1:8000 --require-similar   # any running server
uv run pytest -q tests/test_api_robustness.py
uv run python scripts/api_latency.py   # from a clean src/ tree; writes api_latency_v1
```

Behind a TLS-intercepting proxy set `SMOKE_CA_BUNDLE` (passed as the `ca_bundle` build secret;
with `HTTPS_PROXY` set the build uses `--network=host`). When Docker Hub rate-limits (HTTP 429),
use a mirror:
`SMOKE_BUILD_ARGS="--build-arg NODE_IMAGE=mirror.gcr.io/library/node:22-slim --build-arg PYTHON_IMAGE=mirror.gcr.io/library/python:3.12-slim"`.

## Container runtime (what `container_smoke.sh` checks)

1. The image builds from the current tree.
2. A named volume is seeded from a **copy** of `data/processed` (3 Metrica + the 3 SkillCorner
   matches the e2e flow uses: 1874553, 1899585, 2017461, plus their `*_match.json` metadata for
   team names), without `similarity_*.parquet`. The container runs as documented
   (`-p …:8000 -v <volume>:/data`).
3. The entrypoint builds both similarity indexes before the server starts (logged and present in
   the volume); the server runs as uid 10001, not root.
4. `scripts/smoke_api.py`: health and match list agree; unknown match is 404; per provider: a
   possession replay with consistent array lengths, a frame analysis with pitch control,
   explanations with provenance, temporal context and the provider's own option model, the six
   response policies, a scenario with `marker_follows` that moves at least one defender
   (`simulated`), and `/similar` with the validated k = 50 estimate and the validation summary
   read from `FOOTYMIND_EXPERIMENTS_DIR` inside the image; the web lab at `/` and its JS bundle.
   Every body is parsed strictly (NaN/Infinity fail).
5. The Docker `HEALTHCHECK` reports `healthy`.
6. The Playwright flow (`web/e2e/flow.spec.ts`, 11 tests) passes against the container
   (`E2E_BASE_URL`).

First start on an **empty** volume with `FOOTYMIND_SOURCES=metrica-sample-1` (and, separately,
games 1 + 2) downloads and checksum-verifies the raw files, ingests them, builds the index and
serves. With a single match the similar-moments panel has no neighbours (its own match is
excluded): `completed_share` and `estimate` are null rather than 0.

CI (`docker` job) loads the built image, starts it on an empty volume with Metrica games 1 and 2
(downloaded by the entrypoint, as a user would), runs `smoke_api.py --require-similar` and waits
for the `HEALTHCHECK`. No data is baked into the image or faked; the e2e flow needs SkillCorner
data and stays local.

Startup cost: the server starts only after the index build (measured here on 4 shared vCPU:
36 s for 3 Metrica matches, 22 s for 3 SkillCorner matches, 143 s for all 20 SkillCorner
matches). The `HEALTHCHECK` start period is 300 s so that a full 23-match volume is not
reported unhealthy while it indexes.

## Clean-setup findings (fresh clone, README commands as written)

| Step | Finding | Fix |
|---|---|---|
| `uv run footymind download` | Without ids it fetches **all 23** sources (SkillCorner ~1.8 GB), not the "~180 MB" Metrica games the README promised (891 MB arrived in the first minute). | README lists the three Metrica ids explicitly and states the default. |
| `uv run footymind ingest` | Without ids it ingests every source and **fails** (`FileNotFoundError`, non-zero exit) after the three Metrica bundles when SkillCorner is not downloaded. | Same. |
| Quick start | No similarity-index step: `/similar` is a 503 and the lab's "Similar moments" panel stays empty. | `footymind build-similarity-index --provider metrica` added. |
| `uv run pytest -q` | Passes (Metrica only; SkillCorner tests skip). | none |
| `footymind serve`, `/docs`, README `curl` scenario | Work. `/` is a 404 until `web/dist` is built, as documented. | none |
| `cd web && npm ci && npm run dev` | Works; Vite proxies `/api`. `npm audit` reports 7 moderate issues in dev-only lint dependencies; `--omit=dev` is clean. | none |
| `npm run build` then `footymind serve` | Serves the lab at `/`; `smoke_api.py --require-similar` passes. | none |
| `scripts/reproduce.sh --dry-run` | Ordering bug: `train-option-model` ran **after** the studies whose value model is the packaged option model (Defensive Shadow stability/validation, run value, movement decomposition), on both providers. A from-scratch rebuild that changed the model would leave those artefacts on the old one. | Moved `train-option-model` before them (real bug fix in `scripts/reproduce.sh`). |

Cheap real reproduction: `uv run footymind experiment experiments/configs/pass_outcome_v1.json`
in the fresh clone (revision 83408a7) reproduces every value of the committed
`experiments/results/pass_outcome_v1/results.json` exactly (fold and pooled metrics,
comparisons with their bootstrap intervals, data checksums; 0 differences). The new file is a
superset: the current runner also evaluates 18 model families added after that result was
committed (decision-time, temporal and option families) and records `config.dataset.rules`.
It took 989 s under a load average of about 8, against 92 s recorded for the original run.

Not changed here (outside this change's ownership): `CLAUDE.md` and `web/README.md` repeat the
argument-less `download`/`ingest` commands.

## API robustness (`tests/test_api_robustness.py`)

Defects found and fixed in `src/footymind/api`:

* A NaN or Infinity in a scenario body (accepted by Python's JSON parser) produced a **500**: the
  default 422 handler echoed the input and could not serialise it. The handler now makes the
  error body JSON-safe, and coordinates, `horizon_s`, `follow_fraction`, `pad_s` and the shadow
  `horizon_s` reject non-finite numbers explicitly (`finite_number`).
* Two interventions for the same player were accepted and one silently won: now a 422.
* `/similar?k=` was silently clamped to 1–50: now validated (422 outside 1–50).
* A corrupt or wrongly shaped similarity index, or an unreadable validation results file, gave a
  500: now a 503 that names the file and the rebuild command (the bad index is not cached).
* `completed_share` was **0.0** when no neighbour was listed: now null.
* Scenario summaries reported `n_options_physics_over_0_5 = 0` when the frame had no ball or no
  possession: now null (the root is `spatial.analysis.summary`, outside this change).
* The served option-model info now names its provider and file; a model labelled for another
  provider is refused (503) on analysis and scenarios. With a provider's model missing, analysis
  serves physics options with `p_success_model = null` and scenarios fall back to
  `value_model = "physics"`; Metrica's model is never used for SkillCorner.
* Replays carry `ball_detected` (one entry per frame, false where the provider extrapolated the
  ball or the ball is null; null for Metrica).

Also covered: unknown match/possession/frame (404), invalid frame numbers and parameters (422),
absent players (422), frames without a ball or a possession, every response-policy bound, the
learned policy without a model (422) and its stated forecast horizon, a strict NaN/Infinity
scan over sampled analysis, shadow, replay, similar and scenario responses on both providers,
and identical results from threaded and serial calls across more matches than the bundle cache
holds.

Observed but left as is: absent players and an unobserved ball keep the default provenance label
`observed` with a null position (the convention in `spatial/state.py`); missing velocities are
treated as 0 m/s in the physics model, with a warning in the response.

## Latency

`experiments/results/api_latency_v1/results.json`: in-process, one request at a time, web-lab
defaults, warm caches, 4 shared vCPU with a load average of 8.5 at the start (recorded).

| Endpoint (warm) | Metrica p50 / p95 (ms) | SkillCorner p50 / p95 (ms) |
|---|---|---|
| frame analysis (with pitch-control grid), n = 40 | 66.0 / 80.4 | 75.5 / 108.5 |
| scenario, policy `none`, 24-member ensemble, n = 12 | 462.6 / 512.1 | 526.6 / 599.2 |
| scenario, `marker_follows`, n = 12 | 488.0 / 524.5 | 532.9 / 673.0 |
| similar moments, k = 8 (n = 39 / 40) | 30.7 / 41.6 | 34.4 / 66.4 |

First calls per match (bundle and model loading) took 234–385 ms for analysis and up to 326 ms
for the first similar-moments call (index load).
