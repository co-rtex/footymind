# Current state and session handoff

_Last updated: 2026-11-13 ~22:00 UTC (session 2). Keep this file factual and update it at
every checkpoint._

## Branch and repository
* Working branch: `claude/loving-wright-i0shmp` (pushed to `origin`). `main` is untouched;
  no PR has been opened yet (awaiting the owner's request).
* Commits are authored as Christian Cortez with no assistant trailers (see CLAUDE.md).
* Worktrees (git-ignored or outside the repo): `.claude/worktrees/agent-*` (frontend
  specialist) and `/home/user/fm-backend` (session-2 backend fixes, already merged).
* **Stop-hook note.** The cloud environment runs a *managed* Stop hook
  (`~/.claude/stop-hook-git-check.sh`, registered in the environment's launcher settings, not
  in this repo). It asks for commits to be re-authored as "Claude" so GitHub shows them as
  verified. That conflicts with the owner's authorship rule and is deliberately not done.
  "Unverified" only means the commits are not signed. To silence it, the owner can change the
  environment's hook settings or set up commit signing. Do not rewrite history.

## What works (verified this session)
| Area | Evidence |
|---|---|
| Ingestion of Metrica Games 1–3 (CSV + EPTS) with checksum-pinned sources and quality gates | `uv run footymind download && uv run footymind ingest`; `tests/test_real_metrica.py`, `tests/test_metrica_epts.py` |
| Spatial analytics, option model v1 (JSON, 32 bootstrap members), scenario engine, Defensive Shadow | `tests/test_pitch_control.py`, `test_option_model.py`, `test_api_and_scenario.py`, `test_defensive_shadow.py` |
| Web lab: replay, analysis, scenario editing, deep links, featured scenario | `cd web && npm test` (52), `npx playwright test e2e/flow.spec.ts` (3/3 against the real API) |
| Docker image (API + web; data downloaded to a volume on first start) | built locally (1.03 GB); e2e flow passed against the container |
| All CI gates locally | `scripts/check.sh` (Python: ruff, format, mypy, 62 pytest; web: typecheck, lint, 52 vitest, build) |
| GitHub CI | run 6 (fa8c598) green on all 4 jobs; run 7 failed on `ruff format`, fixed in bcee11f |

## Research status (numbers in CLAIMS_LEDGER.md)
* RQ1 (3 matches): spatial > event-only on every held-out match. Most of the gain is
  teammate geometry. Defender-only adds a small, consistent gain. Interaction features add
  nothing. The failure-mode table shows the largest gain on releases toward no teammate.
* Physics: literature parameters are miscalibrated. Fitting parameters hits implausible
  bounds (structural misspecification), so the option model is the serving probability.
* Defensive Shadow: robust to tracking noise (ρ 0.95 at σ = 0.1 m). It is 2× chance at
  ranking the eventual interceptor, but **not better than nearest-defender**. It is
  experimental and not a rating.
* In progress: `option_model_v2_lead` (lead-pass features), with pre-specified adoption only
  if the paired LOMO CI excludes 0.

## In progress
* Frontend specialist (worktree): an experimental Defensive Shadow layer in the lab.
* Independent methodology/code review agent (read-only). Its findings will be triaged into fixes.

## Next executable actions
1. Integrate the Defensive Shadow UI layer after reviewing its screenshot and tests.
2. Act on the review findings (fix confirmed defects, with regression tests).
3. Adopt or reject option model v2 based on `experiments/results/option_model_v2_lead`.
4. Research: team-level shadow as a pass-failure predictor beyond Model B; direction-aware
   shadow evaluated on held-out data only.
5. Product: show Defensive Shadow and the featured scenario in README screenshots; consider
   a hosted demo (owner decision: Metrica data licence).

## Decisions needing the owner
* Whether to host a public demo serving Metrica-derived positions (no licence file; the
  provider asks for responsible use and attribution).
* Whether to open a PR from this branch to `main`.
