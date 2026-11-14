#!/bin/sh
# Ensure processed bundles in $FOOTYMIND_DATA_DIR can be served, then run the command.
#
# * Empty volume (first start): download (checksum-verified) and ingest FOOTYMIND_SOURCES
#   (default: the three Metrica sample games, ~180 MB). It takes match ids (skillcorner-<id>,
#   ~90 MB each) and the groups metrica / skillcorner / all. A failure here stops the container.
# * Existing volume: re-ingest bundles built by another pipeline version (an image upgrade),
#   and ingest any source in FOOTYMIND_SOURCES (same default) that has no bundle yet, one
#   source at a time. A failure is reported and the server still starts; the affected matches
#   are left out of /api/matches (listed under "unavailable" in /api/health) and answer 503.
# * Then build each provider's similar-moments index if it is missing or no longer matches
#   the bundles (pipeline version, re-ingested bundles).
set -eu
PROCESSED="${FOOTYMIND_DATA_DIR}/processed"
# The same default in both branches, so a first start that failed part-way is completed on the
# next start (the volume is then no longer empty).
SOURCES="${FOOTYMIND_SOURCES:-metrica-sample-1 metrica-sample-2 metrica-sample-3}"
export SOURCES
if [ -z "$(ls -A "${PROCESSED}" 2>/dev/null)" ]; then
  echo "footymind: no processed data in ${FOOTYMIND_DATA_DIR}; downloading (checksum-verified): ${SOURCES}"
  # shellcheck disable=SC2086
  footymind download ${SOURCES}
  # shellcheck disable=SC2086
  footymind ingest ${SOURCES}
else
  TODO="$(python -c 'import os; from footymind.cli import expand_groups
from footymind.data.store import pending_ingest
print(" ".join(pending_ingest(expand_groups(os.environ["SOURCES"].split()))))')"
  if [ -n "${TODO}" ]; then
    echo "footymind: ingesting outdated or missing sources: ${TODO}"
    # one source at a time: a source that cannot be downloaded does not block the others
    for src in ${TODO}; do
      if ! { footymind download "${src}" && footymind ingest "${src}"; }; then
        echo "footymind: ${src}: ingest failed; it is left out of /api/matches until \`footymind ingest ${src}\` succeeds" >&2
      fi
    done
  fi
fi
# The similar-moments index is derived from the processed bundles (Metrica: about 30 s); the
# panel shows a quiet "no index" note until it exists.
for prov in metrica skillcorner; do
  if ls "${PROCESSED}" 2>/dev/null | grep -q "^${prov}-" \
     && ! python -c 'import sys
from footymind.research.similarity import load_index
try:
    ok = load_index(sys.argv[1]) is not None
except Exception as e:
    print(f"footymind: {e}")
    ok = False
sys.exit(0 if ok else 1)' "${prov}"; then
    echo "footymind: building the ${prov} similar-moments index"
    footymind build-similarity-index --provider "${prov}" || echo "footymind: index build failed (non-fatal)"
  fi
done
exec "$@"
