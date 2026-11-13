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
exec "$@"
