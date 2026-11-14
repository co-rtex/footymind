#!/usr/bin/env bash
# Build the Docker image from the current tree, run it the documented way (a named /data volume)
# on a COPY of local processed data WITHOUT the similar-moments indexes, so the entrypoint builds
# them, then check the running container: health, matches, replay, analysis with explanations,
# a scenario with a response policy, /api/response-policies, /similar with the validated estimate
# and validation summary, the web lab at /, and the Playwright flow (web/e2e/flow.spec.ts).
#
#   scripts/container_smoke.sh                 # build + run + API checks + e2e
#   scripts/container_smoke.sh --no-build      # reuse $SMOKE_IMAGE
#   scripts/container_smoke.sh --no-e2e        # API checks only
#   scripts/container_smoke.sh --keep          # leave the container running for inspection
#
# Environment:
#   FOOTYMIND_DATA_DIR  source of data/processed (default: <repo>/data). Never modified.
#   SMOKE_MATCHES       bundles to mount (default: the 3 Metrica games and the 3 SkillCorner
#                       matches the e2e flow uses). All must exist under processed/.
#   SMOKE_IMAGE         image tag (default footymind:smoke); SMOKE_PORT host port (default 18000)
#   SMOKE_CA_BUNDLE     CA file passed as the `ca_bundle` build secret (TLS-intercepting proxy).
#                       With HTTPS_PROXY set, the build uses --network=host and the proxy.
#   SMOKE_BUILD_ARGS    extra `docker build` arguments, e.g. a registry mirror when Docker Hub
#                       rate-limits: "--build-arg NODE_IMAGE=mirror.gcr.io/library/node:22-slim
#                       --build-arg PYTHON_IMAGE=mirror.gcr.io/library/python:3.12-slim"
#   SMOKE_WAIT_S        seconds to wait for /api/health (default 900: the index build runs first)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
BUILD=1 E2E=1 KEEP=0
for a in "$@"; do
  case "$a" in
    --no-build) BUILD=0 ;;
    --no-e2e) E2E=0 ;;
    --keep) KEEP=1 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done
IMAGE="${SMOKE_IMAGE:-footymind:smoke}"
PORT="${SMOKE_PORT:-18000}"
WAIT_S="${SMOKE_WAIT_S:-900}"
SRC="${FOOTYMIND_DATA_DIR:-$REPO/data}"
MATCHES="${SMOKE_MATCHES:-metrica-sample-1 metrica-sample-2 metrica-sample-3 skillcorner-1874553 skillcorner-1899585 skillcorner-2017461}"
TAG="fm-smoke-$$"
SEED="$(mktemp -d)"
step() { echo; echo "== $*"; }

cleanup() {
  local rc=$?
  if [[ $rc -ne 0 ]] && docker inspect "$TAG" >/dev/null 2>&1; then
    echo "-- container logs (last 60 lines)" >&2
    docker logs --tail 60 "$TAG" >&2 || true
  fi
  if [[ $KEEP -eq 1 && $rc -eq 0 ]]; then
    echo "kept: container $TAG on http://127.0.0.1:$PORT, volume $TAG (docker rm -f $TAG; docker volume rm $TAG)"
  else
    docker rm -f "$TAG" >/dev/null 2>&1 || true
    docker volume rm "$TAG" >/dev/null 2>&1 || true
  fi
  rm -rf "$SEED"
  exit $rc
}
trap cleanup EXIT

docker info >/dev/null || { echo "docker daemon not reachable" >&2; exit 1; }

if [[ $BUILD -eq 1 ]]; then
  step "1. build $IMAGE from $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo ' (dirty tree)')"
  args=(-t "$IMAGE")
  if [[ -n "${HTTPS_PROXY:-}" ]]; then
    args+=(--network=host --build-arg "HTTPS_PROXY=$HTTPS_PROXY" --build-arg "NO_PROXY=${NO_PROXY:-}")
  fi
  [[ -n "${SMOKE_CA_BUNDLE:-}" ]] && args+=(--secret "id=ca_bundle,src=$SMOKE_CA_BUNDLE")
  # shellcheck disable=SC2206
  [[ -n "${SMOKE_BUILD_ARGS:-}" ]] && args+=(${SMOKE_BUILD_ARGS})
  docker build "${args[@]}" .
fi

step "2. seed a named volume from a copy of $SRC/processed (no similarity index)"
mkdir -p "$SEED/processed"
for m in $MATCHES; do
  [[ -f "$SRC/processed/$m/manifest.json" ]] || { echo "missing bundle $SRC/processed/$m" >&2; exit 1; }
  cp -r "$SRC/processed/$m" "$SEED/processed/"
  # SkillCorner team names and dates come from the provider's small match metadata file.
  if [[ $m == skillcorner-* && -f "$SRC/raw/skillcorner/${m#skillcorner-}_match.json" ]]; then
    mkdir -p "$SEED/raw/skillcorner"
    cp "$SRC/raw/skillcorner/${m#skillcorner-}_match.json" "$SEED/raw/skillcorner/"
  fi
done
if ls "$SEED/processed"/similarity_* >/dev/null 2>&1; then echo "seed holds an index" >&2; exit 1; fi
chmod -R a+rX "$SEED"
docker volume create "$TAG" >/dev/null
docker run --rm --entrypoint sh -v "$TAG:/data" -v "$SEED:/seed:ro" "$IMAGE" -c 'cp -r /seed/. /data/'

step "3. run the container (documented configuration: -p ...:8000 -v <volume>:/data)"
docker run -d --name "$TAG" -p "127.0.0.1:$PORT:8000" -v "$TAG:/data" "$IMAGE" >/dev/null
uid=$(docker exec "$TAG" id -u)
[[ $uid != 0 ]] || { echo "container runs as root" >&2; exit 1; }
deadline=$((SECONDS + WAIT_S))
until curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null; do
  [[ "$(docker inspect -f '{{.State.Running}}' "$TAG")" == true ]] || { echo "container exited" >&2; exit 1; }
  (( SECONDS < deadline )) || { echo "no health after $WAIT_S s" >&2; exit 1; }
  sleep 3
done
logs=$(docker logs "$TAG" 2>&1)
for prov in metrica skillcorner; do
  if grep -q " ${prov}-" <<<" $MATCHES"; then
    grep -q "building the ${prov} similar-moments index" <<<"$logs" \
      || { echo "entrypoint did not build the $prov index" >&2; exit 1; }
    docker exec "$TAG" test -s "/data/processed/similarity_${prov}.parquet" \
      || { echo "no $prov index in the volume" >&2; exit 1; }
  fi
done
if grep -q "index build failed" <<<"$logs"; then echo "an index build failed" >&2; exit 1; fi
echo "healthy; entrypoint built the indexes; user uid $uid"

step "4a. API and web checks"
# shellcheck disable=SC2086
python3 scripts/smoke_api.py "http://127.0.0.1:$PORT" --expect-matches $MATCHES --require-similar

step "4b. Docker HEALTHCHECK reports healthy"
until [[ "$(docker inspect -f '{{.State.Health.Status}}' "$TAG")" == healthy ]]; do
  (( SECONDS < deadline )) || { echo "HEALTHCHECK never reported healthy" >&2; exit 1; }
  sleep 3
done
echo healthy

if [[ $E2E -eq 1 ]]; then
  step "4c. Playwright flow against the container"
  [[ -d web/node_modules ]] || (cd web && npm ci --no-audit --no-fund)
  (cd web && E2E_BASE_URL="http://127.0.0.1:$PORT" npx playwright test e2e/flow.spec.ts --trace off)
fi
echo
echo "container smoke passed ($IMAGE)"
