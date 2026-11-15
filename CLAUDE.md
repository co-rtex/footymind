# CLAUDE.md: working agreement for agents in this repository

Read `docs/CURRENT_STATE.md` first. It holds the current branch, what works, the last
verified results and the next action.

## Commands
```bash
uv sync                                   # Python env (3.11+; developed on 3.13)
uv run footymind download metrica         # 3 Metrica games, ~180 MB, checksum-verified
uv run footymind ingest metrica           # ~3 s/match
#   (`all` = also the 20 SkillCorner matches, ~2 GB; with no argument both commands stop)
uv run footymind build-similarity-index   # per-provider "Similar moments" index (derived)
uv run pytest -q                          # all tests; real-data tests skip if data is absent
scripts/check.sh                          # every CI gate locally (Python + web); run before committing
uv run footymind serve                    # API on :8000 (serves web/dist at / if built)
uv run footymind experiment experiments/configs/pass_outcome_v3.json
scripts/reproduce.sh --dry-run            # the committed research artefacts, in dependency order
cd web && npm ci && npm run dev           # frontend on :5173 (proxies /api to the API)
```

## Non-negotiable rules
- Never fabricate observations. Unobserved = NaN/null. Synthetic fixtures must say so.
- Every analytical output carries a `Provenance` label (observed, derived, estimated,
  hypothetical, simulated). Keep these labels visible in the API and the UI.
- Decision-time model features: decision-frame positions, **causal** velocities, the
  release direction/speed truncated before the outcome frame, and prior events only. Never
  use event end locations, durations, receiver ids or outcome subtypes (see ADR-0005).
  On broadcast tracking the event start frame often already contains the kick: decision-time
  claims for SkillCorner use the 0.2 s guard frame (`static_offset_s`/`temporal_offset_s`,
  ADR-0007). "No later frame is read" is not a sufficient leakage test. Also test at the guard.
- Unobserved or censored inputs stay NaN (e.g. time-on-ball cut short by an unobserved ball);
  never turn "unknown" into 0.
- Evaluation: leave-one-match-out, a-priori hyperparameters, and possession-clustered CIs.
  Any number quoted in README/docs must have an entry in `docs/CLAIMS_LEDGER.md`
  pointing to a committed `experiments/results/*/results.json` or a test.
- Do not commit raw or processed data (`/data` is ignored). Do not commit model pickles.
- Counterfactual outputs are model-based comparisons, never causal claims.
- Bump `data.store.PIPELINE_VERSION` when the processed representation changes meaning.
- Never pool providers (Metrica vs SkillCorner) in a model or evaluation; scope research loops
  with `startswith(provider + "-")` and keep per-provider option models.

## Git
- Work on feature branches. Never push to `main`. Commits are authored as the repository
  owner (Christian Cortez). Commit messages describe the engineering change only, with no
  AI/assistant attribution or co-author trailers.
- Run `scripts/check.sh` before committing and gate on its **exit code** (never pipe it through
  `grep`/`tail` in an `&&` chain: that masks failures; it happened once on 2026-11-13).
- Run lint, types and tests before committing. Re-run experiments from a clean revision
  before committing their results (the runner records `code_revision`).

## Layout
`src/footymind/{data,tracking,spatial,counterfactual,research,api}`, `web/` (frontend),
`experiments/{configs,results}`, `docs/` (spec, architecture, ADRs, research, claims, state),
`legacy/` (archived v1 predictor).
