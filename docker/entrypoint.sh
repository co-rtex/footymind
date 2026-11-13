#!/bin/sh
# Ensure processed Metrica bundles exist in $FOOTYMIND_DATA_DIR, then run the command.
set -eu
if [ -z "$(ls -A "${FOOTYMIND_DATA_DIR}/processed" 2>/dev/null)" ]; then
  echo "footymind: no processed data in ${FOOTYMIND_DATA_DIR}; downloading Metrica sample data (checksum-verified)"
  footymind download
  footymind ingest
fi
exec "$@"
