# Current state and session handoff

_Last updated: 2026-11-14, session 3 (started 01:36:18 UTC). Keep this file factual and update
it at every checkpoint. The session-3 handoff section at the end is filled in at close._

## Branch and repository
* Working branch: `claude/loving-wright-i0shmp` (pushed to `origin`). `main` is untouched, and
  no PR has been opened (the owner has not asked for one).
* Commits are authored as Christian Cortez with no assistant trailers (see CLAUDE.md).
* Specialist worktrees under `.claude/worktrees/` (agent branches `worktree-agent-*`) are
  merged into this branch and can be removed. Scratch experiment worktrees live outside the
  repo, in the session scratchpad.
* **Stop-hook note.** The cloud environment runs a *managed* Stop hook
  (`~/.claude/stop-hook-git-check.sh`) that asks for commits to be re-authored as "Claude".
  That conflicts with the owner's authorship rule and is deliberately not done. Do not rewrite
  history.

## Data (local, not committed)
`data/raw` holds Metrica (about 180 MB) and SkillCorner (about 1.8 GB). `data/processed`
holds 23 bundles at **pipeline version 2**, plus the derived similarity indexes
`similarity_{metrica,skillcorner}.parquet` (`footymind build-similarity-index`).
SkillCorner bundles ingested before `ac0629f` have no `ball_detected` mask. Re-ingest them
(`uv run footymind ingest <ids>`) so that an extrapolated ball is labelled *estimated*.
Positions are unchanged, so committed results stay valid.

## What exists (session 3 additions in bold)
* Ingestion (Metrica CSV/EPTS, SkillCorner v2), canonical model with provenance, quality gates.
* Spatial analytics, option model (per provider), Possession Multiverse.
* **Opponent response policies** (`counterfactual/response.py`, `kinematics.py`, ADR-0006):
  `none`, `marker_follows`, `press_carrier`, `zonal_shift`, `lane_cover`, `learned`. The
  learned policy is validated as a short-horizon forecast of real play (C40–C44). The rule
  policies are experimental. **An observational check of the rules is in
  `research/rule_response_check.py`.**
* **Causal temporal context** (`spatial/temporal.py`): time-on-ball (censored), last-second
  movement history, guard frames (ADR-0007). Served with the frame analysis.
* **Evidence-grounded explanations** (`spatial/explain.py`) and **similar-moment retrieval**
  (`research/similarity.py`, `GET .../similar`), both shown in the lab.
* **Attack vs defense movement decomposition** (`ghost/movement_decomposition.py`, C31/C32).
* **Data-quality audit** (`research/data_quality_audit.py`, C30).
* **Match facts** for SkillCorner (team short names, competition, round, date; no player
  names).
* Web lab: response-policy selector with static-vs-response comparison, "Why this moment",
  "Similar moments", product-audit fixes (`docs/product/AUDIT_SESSION3.md`), Vitest 5.
