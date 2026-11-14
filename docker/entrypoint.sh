#!/bin/sh
# Ensure processed bundles in $FOOTYMIND_DATA_DIR can be served, then run the command.
#
# * Empty volume (first start): download (checksum-verified) and ingest FOOTYMIND_SOURCES
#   (default: the three Metrica sample games, ~180 MB). Add SkillCorner match ids
#   (skillcorner-<id>, ~90 MB each) if wanted. A failure here stops the container.
# * Existing volume: re-ingest bundles built by another pipeline version (an image upgrade),
#   and ingest any source named in FOOTYMIND_SOURCES that has no bundle yet. A failure here
#   is reported and the server still starts; the affected matches answer 503 naming the fix.
# * Then build each provider's similar-moments index if it is missing or no longer matches
#   the bundles (pipeline version, re-ingested bundles).
set -eu
PROCESSED="${FOOTYMIND_DATA_DIR}/processed"
if [ -z "$(ls -A "${PROCESSED}" 2>/dev/null)" ]; then
  SOURCES="${FOOTYMIND_SOURCES:-metrica-sample-1 metrica-sample-2 metrica-sample-3}"
  echo "footymind: no processed data in ${FOOTYMIND_DATA_DIR}; downloading (checksum-verified): ${SOURCES}"
  # shellcheck disable=SC2086
  footymind download ${SOURCES}
  # shellcheck disable=SC2086
  footymind ingest ${SOURCES}
else
  TODO="$(python -c 'import os; from footymind.data.store import pending_ingest
print(" ".join(pending_ingest(os.environ.get("FOOTYMIND_SOURCES", "").split())))')"
  if [ -n "${TODO}" ]; then
    echo "footymind: ingesting outdated or missing sources: ${TODO}"
    # shellcheck disable=SC2086
    if ! { footymind download ${TODO} && footymind ingest ${TODO}; }; then
      echo "footymind: ingest failed; those matches will answer 503 until \`footymind ingest\` succeeds" >&2
    fi
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
