#!/bin/sh
# Ensure processed bundles exist in $FOOTYMIND_DATA_DIR, then run the command.
# FOOTYMIND_SOURCES selects which registered sources to fetch on first start (default: the three
# Metrica sample games, ~180 MB). Add SkillCorner match ids (skillcorner-<id>, ~90 MB each) if wanted.
set -eu
SOURCES="${FOOTYMIND_SOURCES:-metrica-sample-1 metrica-sample-2 metrica-sample-3}"
if [ -z "$(ls -A "${FOOTYMIND_DATA_DIR}/processed" 2>/dev/null)" ]; then
  echo "footymind: no processed data in ${FOOTYMIND_DATA_DIR}; downloading (checksum-verified): ${SOURCES}"
  # shellcheck disable=SC2086
  footymind download ${SOURCES}
  # shellcheck disable=SC2086
  footymind ingest ${SOURCES}
fi
# The similar-moments index is derived from the processed bundles. Build it once per provider
# present (Metrica: about 30 s); the panel shows a quiet "no index" note until it exists.
for prov in metrica skillcorner; do
  if ls "${FOOTYMIND_DATA_DIR}/processed" 2>/dev/null | grep -q "^${prov}-" \
     && [ ! -f "${FOOTYMIND_DATA_DIR}/processed/similarity_${prov}.parquet" ]; then
    echo "footymind: building the ${prov} similar-moments index"
    footymind build-similarity-index --provider "${prov}" || echo "footymind: index build failed (non-fatal)"
  fi
done
exec "$@"
