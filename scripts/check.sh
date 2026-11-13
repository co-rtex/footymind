#!/usr/bin/env bash
# Run the same quality gates as CI (.github/workflows/ci.yml) locally, before committing.
#   scripts/check.sh          # Python + web
#   scripts/check.sh python   # Python only
set -euo pipefail
cd "$(dirname "$0")/.."
what="${1:-all}"
if [[ "$what" == "all" || "$what" == "python" ]]; then
  uv run ruff check src tests
  uv run ruff format --check src tests
  uv run mypy
  uv run pytest -q
fi
if [[ "$what" == "all" || "$what" == "web" ]]; then
  (cd web && npm run typecheck && npm run lint && npm test && npm run build)
fi
echo "all checks passed ($what)"
