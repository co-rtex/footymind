# CLAUDE.md: working agreement for agents in this repository

Read `docs/CURRENT_STATE.md` first. It holds the current branch, what works, the last
verified results and the next action.

## Commands
```bash
uv sync                                   # Python env (3.11+; developed on 3.13)
uv run footymind download                 # fetch Metrica sample data (checksum-verified, ~180 MB)
uv run footymind ingest                   # build data/processed/<match>/ bundles (~3 s/match)
uv run pytest -q                          # all tests; real-data tests skip if data is absent
uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy
uv run footymind serve                    # API on :8000 (serves web/dist at / if built)
uv run footymind experiment experiments/configs/pass_outcome_v1.json
cd web && npm ci && npm run dev           # frontend on :5173 (proxies /api to the API)
```

## Non-negotiable rules
- Never fabricate observations. Unobserved = NaN/null. Synthetic fixtures must say so.
- Every analytical output carries a `Provenance` label (observed, derived, estimated,
  hypothetical, simulated). Keep these labels visible in the API and the UI.
- Decision-time model features: decision-frame positions, **causal** velocities, the
  release direction/speed truncated before the outcome frame, and prior events only. Never
  use event end locations, durations, receiver ids or outcome subtypes (see ADR-0005).
- Evaluation: leave-one-match-out, a-priori hyperparameters, and possession-clustered CIs.
  Any number quoted in README/docs must have an entry in `docs/CLAIMS_LEDGER.md`
  pointing to a committed `experiments/results/*/results.json` or a test.
- Do not commit raw or processed data (`/data` is ignored). Do not commit model pickles.
- Counterfactual outputs are model-based comparisons, never causal claims.
- Bump `data.store.PIPELINE_VERSION` when the processed representation changes meaning.

## Git
- Work on feature branches. Never push to `main`. Commits are authored as the repository
  owner (Christian Cortez). Commit messages describe the engineering change only, with no
  AI/assistant attribution or co-author trailers.
- Run lint, types and tests before committing. Re-run experiments from a clean revision
  before committing their results (the runner records `code_revision`).

## Layout
`src/footymind/{data,tracking,spatial,counterfactual,research,api}`, `web/` (frontend),
`experiments/{configs,results}`, `docs/` (spec, architecture, ADRs, research, claims, state),
`legacy/` (archived v1 predictor).
