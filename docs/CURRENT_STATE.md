# Current state and session handoff

_Last updated: 2026-11-13 (end of session 1, part 1). Keep this file factual. Update it at
every checkpoint._

## Branch and repository
* Working branch: `claude/loving-wright-i0shmp` (pushed to `origin`). `main` is untouched.
* Commits are authored as Christian Cortez with no assistant trailers (see CLAUDE.md).
* Agent worktrees live under `.claude/worktrees/` (git-ignored).

## What works (verified)
| Area | Evidence |
|---|---|
| Metrica Games 1–2 download (SHA-256 pinned) and ingestion | `uv run footymind download metrica-sample-1 metrica-sample-2 && uv run footymind ingest` (~3 s/match) |
| Data-quality gates (sync, ids, frames) | manifests in `data/processed/*/manifest.json` |
| Spatial analytics + scenario engine + API | `tests/test_pitch_control.py`, `tests/test_api_and_scenario.py` |
| Research RQ1 v1 + ablation + 2 sensitivity runs | `experiments/results/pass_outcome_v1*/report.md` |
| Python quality gates | `uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv run mypy`, `uv run pytest -q` (41 passed at last run) |
| Legacy v1 tests | `cd legacy/match_outcome_predictor && PYTHONPATH=. pytest -q tests` (4 passed) |

## In progress
* **Frontend** (`web/`): built by a frontend specialist in an isolated worktree. It will be
  merged after review of screenshots and e2e results. Until then `web/` does not exist on
  this branch.
* **CI** (`.github/workflows/ci.yml`): written. Not yet observed running on GitHub.

## Next executable actions (priority order)
1. Merge and review the frontend. Add a web job to CI. Run the e2e flow against the real API.
2. Metrica Game 3 (EPTS) adapter: `tracking.txt` + `metadata.xml` + `events.json`.
   Subtypes are lists. `CARRY` events exist (absent in games 1–2): exclude them from
   event-history features to avoid a provider-format shift. Cross-check the inferred attack
   direction against the metadata (`attack_direction_first_half`). Then re-run RQ1 with 3 folds.
3. Fit the PPCF parameters on training folds (calibrated physics). Compare with the literature parameters.
4. Defensive Shadow v1 (see PROJECT_SPEC roadmap item 5).

## Known issues and limitations
* Physics estimates (p_success) are uncalibrated and overconfident (ECE 0.18 on observed passes).
* Lane model: the uniform reaction time under-weights defenders standing in short lanes (documented and tested).
* Carrier detection uses proximity (≤ 1.5 m) with an event fallback (±3 frames).
* Possession rules are v1 and simple (documented in DATA_SOURCES.md).
* Metrica raw data has no explicit licence: attribute it, and do not redistribute raw files.

## Decisions needing the owner
* Whether to publish a hosted demo that serves Metrica-derived positions (the provider
  asks for responsible use and attribution; there is no licence file).
* Opening a PR from this branch to `main` (not done; awaiting the owner's request).
