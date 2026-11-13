# Current state and session handoff

_Last updated: 2026-11-13 ~22:45 UTC (session 2, which started 21:34 UTC). Keep this file
factual and update it at every checkpoint._

## Branch and repository
* Working branch: `claude/loving-wright-i0shmp` (pushed to `origin`). `main` is untouched,
  and no PR has been opened (the owner has not asked for one).
* Commits are authored as Christian Cortez with no assistant trailers (see CLAUDE.md).
* Worktrees (ignored or outside the repo): `.claude/worktrees/agent-*` (frontend
  specialist, fully merged) and `/home/user/fm-backend` (branch `session2-backend-fixes`,
  fully merged). Both can be removed with `git worktree remove`.
* **Stop-hook note.** The cloud environment runs a *managed* Stop hook
  (`~/.claude/stop-hook-git-check.sh`, registered in the environment's launcher settings, not
  in this repo, and protected from edits). It asks for commits to be re-authored as "Claude"
  so GitHub shows them as verified. That conflicts with the owner's authorship rule and is
  deliberately not done. "Unverified" only means the commits are unsigned. The owner can
  change the environment's hook settings or set up commit signing. Do not rewrite history.
* A background `dockerd` was started for the image build test. Stop it with `pkill dockerd`
  if it is still running.

## What works (verified)
| Area | Evidence / command |
|---|---|
| Ingestion: Metrica Games 1–3 (CSV + EPTS) and **20 SkillCorner A-League matches** (broadcast, 10 fps), with checksum-pinned sources, quality gates and attacking direction cross-checked with provider metadata | `uv run footymind download [ids]`, `uv run footymind ingest [ids]`; tests `test_real_metrica.py`, `test_metrica_epts.py`, `test_skillcorner_adapter.py` (synthetic, runs in CI), `test_real_skillcorner.py` |
| Provenance for broadcast data: extrapolated (off-camera) positions are `estimated` in the API and drawn hollow and dashed in the UI; absent players are never "estimated" | `test_real_skillcorner.py`; e2e `broadcast tracking ...` |
| Spatial analytics, **provider-specific option models** (JSON with 32 bootstrap members; never pooled across providers), scenario engine (final-state feasibility, checked simulated moves), Defensive Shadow, run value | `tests/test_option_model.py`, `test_api_and_scenario.py`, `test_defensive_shadow.py`, `test_run_value.py` |
| Web lab: replay, analysis, scenarios, deep links, featured scenario, Defensive Shadow layer, 23-match landing grouped by provider | `cd web && npm test` (73), e2e `npx playwright test e2e/flow.spec.ts` (5/5 against the real API and the FastAPI-served build) |
| Docker image (API + web; data downloaded to a volume on first start, `FOOTYMIND_SOURCES`) | built locally (1.03 GB); e2e passed against the container; CI `docker` job |
| All CI gates locally | `scripts/check.sh` (exit code!): Python ruff, format, mypy, 72+ pytest; web typecheck, lint, 73 vitest, build |
| GitHub CI | green on 89c1d08 (run 14), including the Docker job; data job pinned to Metrica |
| API latency | `/api/matches` 15 ms (was 2.5 s); frame analysis about 0.09 s warm; scenario about 0.3–0.7 s; shadow about 0.4–0.7 s |

## Research status (all numbers in CLAIMS_LEDGER.md)
* **Main, replicated result (C20, C23):** decision-time spatial context improves held-out
  pass-outcome prediction. Metrica Δ log loss −0.026 [−0.034, −0.018]; SkillCorner −0.024
  [−0.027, −0.020], better in 20/20 held-out matches. The drivers are defensive structure
  relative to the ball and attacking width.
* Action-conditioned gain: −0.071 (Metrica) and −0.044 (SkillCorner). This is
  provider-dependent and is partly execution quality.
* Interaction features: null on 3 matches, small and significant (−0.004) on 20 matches.
* Physics: literature parameters are miscalibrated, and fitting them hits implausible bounds
  (structural misspecification). The option model serves probabilities.
* Defensive Shadow: robust to noise. It finds the eventual interceptor at 2× chance but is not
  better than nearest-defender. Experimental.
* Run value: small positive association with being chosen. Weak as a ranking. Experimental.
  A SkillCorner replication was running at handoff (`run_value_validation_v1_skillcorner`).
* An independent methodology review (2026-11-13) found no hard leakage. Its fixes are
  rules v2, decision-time framing and scenario fixes. Rules v1 still reproduce the old results.

## Next executable actions
1. Record the SkillCorner run-value replication (if not already committed).
2. Research: a learned defender-response policy (ghosting) is required before off-ball
   valuation can be credible; test team-level shadow as a pass-failure predictor beyond
   Model B on SkillCorner.
3. Product: a provider-aware match clock in the API (`period_time_s`), and SkillCorner event
   positions (scaled) for event markers.
4. Hygiene: remove merged worktrees; consider a Vitest 5 upgrade (dev-only audit findings).

## Decisions needing the owner
* Whether to host a public demo serving provider-derived positions (Metrica has no licence
  file; SkillCorner is MIT with a credit request).
* Whether to open a PR from this branch to `main`.
* Whether to change the managed stop hook or enable commit signing.
