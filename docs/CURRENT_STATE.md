# Current state and session handoff

_Last updated: 2026-11-14 03:34 UTC, end of session 3 (01:36:18–03:3x UTC). Keep this file
factual and update it at every checkpoint._

## Branch and repository
* Working branch: `claude/loving-wright-i0shmp` (pushed to `origin`). `main` is untouched, and
  no PR has been opened (the owner has not asked for one).
* Commits are authored as Christian Cortez with no assistant trailers (see CLAUDE.md).
* The specialist worktrees and their branches were merged and removed. Scratch experiment
  worktrees (detached, outside the repo, in the session scratchpad) can be pruned with
  `git worktree prune` once the scratchpad is gone.
* **Stop-hook note.** The cloud environment runs a *managed* Stop hook
  (`~/.claude/stop-hook-git-check.sh`) that asks for commits to be re-authored as "Claude".
  That conflicts with the owner's authorship rule and is deliberately not done. Do not rewrite
  history.

## Data (local, not committed)
`data/raw` holds Metrica (about 180 MB) and SkillCorner (about 1.8 GB). `data/processed`
holds 23 bundles at **pipeline version 2**, plus the derived similarity indexes
`similarity_{metrica,skillcorner}.parquet` (`footymind build-similarity-index`).
All 20 SkillCorner bundles were re-ingested at the end of session 3 and carry the
`ball_detected` mask. Positions are unchanged, so committed results stay valid. Bundles
ingested before `ac0629f` load the mask as None.

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

## Verification (end of session 3)
| Gate | Result |
|---|---|
| `scripts/check.sh` (exit code 0) | ruff, format, mypy, **166 pytest**; web typecheck, lint, **130 vitest** (Vitest 5.1.0), build |
| `cd web && npx playwright test --trace off` | **16/16** (11 flow tests + 5 screenshot tests) against the real API, also re-run after re-ingesting SkillCorner |
| GitHub CI | green for every session-3 push checked (runs 28–33). Later runs were in progress at handoff; the `docker` job now copies `experiments/results` into the image |
| SkillCorner bundles | re-ingested at the end of the session with `ball_detected`. An extrapolated ball is now *estimated* in the API and explanations |

## Research status (numbers live in CLAIMS_LEDGER.md; session 3 = C30–C44)
* **RQ2, temporal context (C33–C35, C38).** At the 0.2 s guard frame (ADR-0007), 1 s of movement
  history improves decision-time pass prediction beyond static spatial context and time-on-ball:
  Metrica −0.014 [−0.021, −0.007], SkillCorner −0.009 [−0.011, −0.007] (20/20 matches,
  match-level CI). The gain is mainly discrimination. Time-on-ball is small (SkillCorner −0.002;
  not robust on Metrica), and event timing alone is null. **C20/C28 are unchanged under the
  guard.**
* **Kick leakage on broadcast frames (C38).** At the event start frame the ball is already
  leaving the passer in 62.7% of SkillCorner passes (1.8% Metrica). By −0.2 s the share reaches
  the noise floor, so decision-time SkillCorner claims use the guard.
* **Off-ball credit (C31/C32, revised).** Attack and defense movement roughly cancel on average.
  The large interaction is mostly the option model's non-additivity (rotation placebo 0.51 /
  0.53 vs 0.58 / 0.64), with a small genuine coupling on SkillCorner. Per-pass run credit depends
  on the convention.
* **Response policies (C37 revised, C40–C44).** `learned` forecasts real play better than constant
  velocity but is nearly inert. Under the innovation design, `marker_follows` and `lane_cover`
  match real reactions on both providers (β ≈ 1.0–1.3, beyond a placebo). `press_carrier`
  passes on Metrica only, and `zonal_shift` overshoots. The served assumptions state this per
  policy.
* **Similar moments (C36).** A k = 50 shrunk neighbour rate beats the base rate held-out on both
  providers. The API serves that exact estimate next to the 8 listed neighbours.
* **Defensive Shadow (C29).** At chance on all 20 SkillCorner matches; worse than
  nearest-defender in 18/20.
* Two independent methodology reviews were run this session. Every major finding was fixed
  or reported (see the RQ2 amendment and the "revised" ledger rows).

## Next executable actions (prioritised)
1. **Off-ball value with a validated response.** Combine the innovation-validated rules
   (`marker_follows`, `lane_cover`) or a richer learned model (conditioned on attacker
   displacements) with a possession-value target. Re-run the decomposition with the placebo as
   the null.
2. **Response envelope v2.** The a-priori envelope contains only 61–73% of real 1 s positions
   (C41, C43). Pre-register a v2 (shorter reaction delay) and re-validate.
3. **Time-on-ball on broadcast.** Censor runs at *undetected* ball frames now that the mask exists.
   Re-run `temporal_guard_skillcorner_v1_02s` (expect a small change to H-T1 only).
4. **Similar moments.** Pre-register and validate the displayed k = 8 list (Brier score) or
   display k = 50.
5. **Docker.** Build the image locally once with the new entrypoint (similarity index on first
   start) and `FOOTYMIND_EXPERIMENTS_DIR`; CI builds it but does not run the container.

## Decisions needing the owner
* Whether to host a public demo (Metrica has no licence file; SkillCorner is MIT with a
  credit request, and the app now shows SkillCorner team names, competition and date).
* Whether to open a PR from this branch to `main`.
* Whether to change the managed stop hook or enable commit signing.

## Resume
```bash
git checkout claude/loving-wright-i0shmp && uv sync && (cd web && npm ci)
uv run footymind download && uv run footymind ingest && uv run footymind build-similarity-index
scripts/check.sh && (cd web && npx playwright test --trace off)
scripts/reproduce.sh --dry-run
```
