# Current state and session handoff

_Last updated: 2026-11-14, end of session 4 (overnight completion session, started 03:47:03 UTC).
Keep this file factual and update it at every checkpoint._

## Branch and repository
* Working branch: `claude/loving-wright-i0shmp` (pushed to `origin`). `main` is untouched, and
  no PR has been opened (the owner has not asked for one).
* Commits are authored as Christian Cortez with no assistant trailers (see CLAUDE.md).
* All specialist worktrees and branches from session 4 were merged and removed. A detached
  scratch worktree in the session scratchpad (`wt4`) was used only to run experiments from a
  clean revision; `git worktree prune` removes its record once the scratchpad is gone.
* **Stop-hook note.** The cloud environment runs a *managed* Stop hook
  (`~/.claude/stop-hook-git-check.sh`) that asks for commits to be re-authored as "Claude".
  That conflicts with the owner's authorship rule and is deliberately not done. Do not rewrite
  history.

## Data (local, not committed)
`data/raw` holds Metrica (about 180 MB) and SkillCorner (about 1.8 GB). `data/processed`
holds 23 bundles plus the derived similarity indexes `similarity_{metrica,skillcorner}.parquet`.

**Pipeline version 3** (session 4): SkillCorner event and possession ends stay inside their
period (17 possessions, one per affected match, used to end in the next half). Bundles from
another version are refused with a 503 naming `footymind ingest <id>`; similarity indexes record
the pipeline version and bundle timestamps and are refused when stale. After pulling, re-ingest
and rebuild:

```bash
uv run footymind ingest            # all 23 sources if SkillCorner is downloaded
uv run footymind build-similarity-index
```

Pass-dataset rows and values are identical at v2 and v3 (19,507 rows; 19,485 at the 0.2 s
guard), so research results committed on v2 bundles are unaffected. The data-quality audit was
re-run at v3 (C30). The Docker entrypoint re-ingests outdated bundles and rebuilds stale
indexes on start.

## What exists
* Ingestion (Metrica CSV/EPTS at 25 Hz, SkillCorner v2 broadcast at 10 fps), canonical model with
  provenance, quality gates, possession table, frame-rate-safe time constants (0.3 s
  event-carrier tolerance).
* Spatial analytics, per-provider calibrated option models, Possession Multiverse.
* **Opponent response policies** (`counterfactual/response.py`, ADR-0006): `none`,
  `marker_follows`, `press_carrier`, `zonal_shift`, `lane_cover`, `learned` (forecaster, nearly
  inert), `learned_innovation` (observational co-movement model, session 4). Each policy carries
  a `status` sentence and a machine-readable `verdict` (`observational_supported`,
  `not_supported`, `forecast_validated`, `observational_validated`, `unchecked`,
  `no_response`). Kinematic **envelope v2** clips every simulated move.
* Causal temporal context, guard frames (ADR-0007, calibrated, C38), evidence-grounded
  explanations, similar-moment retrieval with the validated k = 50 estimate (C36).
* Research modules: rule response check v4 (paired, rotation placebo, C70), off-ball opportunity
  under reactive defenders (`ghost/offball_opportunity.py`, `research/offball_study.py`),
  guard calibration and ball fallback (C38), movement decomposition (C31/C32), Defensive Shadow
  validation (C17/C29, status served per provider).
* Web lab: guided tour with three featured situations (filtered to the matches served),
  policy selector with verdict badges and the learned-innovation fitted-range warning,
  API-sourced reach and shadow parameters, 30 s request timeout, similar moments, phone layout.
* API hardening: strict JSON (no NaN/Infinity), 422 bodies without the input echoed, 1 MB
  request-body cap (413), cache freshness after re-ingest.
* Docker image verified at runtime (CI `docker` job: empty volume, Metrica 1–2 downloaded by
  the entrypoint, `smoke_api.py --require-similar`, `HEALTHCHECK`).

## Verification (end of session 4)
| Gate | Result |
|---|---|
| `scripts/check.sh` (exit code 0, all 23 bundles at v3) | ruff, format, mypy, **339 pytest passed, 0 skipped**; web typecheck, lint, **211 vitest**, build |
| `cd web && npx playwright test --trace off` | **27/27** against the real API on v3 data (flow, screenshots, broadcast tests) |
| GitHub CI | every job green for 83d4830, including the container-runtime `docker` job; see the final report for the newest run |
| `scripts/reproduce.sh --dry-run` | exit 0, every committed artefact in dependency order |
| Container | built and run in CI on every push; locally verified in session 4 by the release-engineering agent (Playwright 11/11 against the container) |

## Research status (numbers live in CLAIMS_LEDGER.md; session 4 = C45–C50, C55–C58, C61–C71)
* **RQ1/RQ2 unchanged and strengthened.** Spatial context improves decision-time pass prediction
  on both providers (C20, C28), also with every feature read 0.2 s before the release (C34,
  C35); the last second of movement adds a further gain at the guard frame (C33, C35).
  Time-on-ball censoring at extrapolated balls does not change the temporal findings (C50).
  Guarded option models (C71).
* **Guard calibration (C38).** The SkillCorner kick-underway share is 0.627 at the event start
  frame and reaches its floor by −0.2 s. Ball fallback at the guard is rare (Metrica 68 open-play
  passes of 3,656, SkillCorner 15 of 16,006).
* **Kinematic envelope v2 (C55–C58).** No reaction delay, speed-dependent acceleration, 9 m/s
  top speed. It covers 0.998–1.000 of real defender positions at 0.5–1.5 s (v1 0.19–0.86) and
  still rejects real positions shifted 5 m at 0.5–1.0 s (0.96–1.00), with a larger area (about
  40 m² at 1.0 s against 3 m²). At the 1.5 s scenario default most 5 m shifts are reachable.
* **Learning responses (C61, C62).** Positions barely predict defender deviations, so the
  forecaster is inert; an innovation model conditioned on attackers' unexpected movement beats
  constant velocity and a placebo on both providers and is served as `learned_innovation`
  (observational, not causal). Response sizes under envelope v2: C63.
* **Serving (C65–C69).** Container runtime, first start on an empty volume, API latency,
  robustness (no NaN/Infinity, 422/404/503 contracts), per-provider model isolation.
* **Rule check v4 (C70).** `marker_follows` and `lane_cover` beat constant velocity and a
  size-matched rotation placebo on both providers (paired, possession-clustered,
  match-stratified). `press_carrier` and `zonal_shift` do not; the lab labels them
  "not supported".
* **Off-ball value under reactive defenders (C45–C49, re-run under envelope v2 at a3c88d2).**
  Real runs beat randomly directed ones (+0.0042 Metrica, +0.0037 SkillCorner). Putting back
  defenders' reactions changes the credit (best progressive option) by at most 0.0003 on average (`lane_cover` non-zero
  but negative); a fitted response moving 8–9 defenders changes nothing either (positive
  control, post-review). The credit depends on the constant-velocity vs observed background,
  and no adjusted variant beats static on the outcome checks. Not exposed in the product
  (`docs/research/offball_opportunity.md`).
* **Similar moments (C36)**, **Defensive Shadow (C17, C29)**, **movement decomposition
  (C31, C32)** as before.
* Session 4 ran one independent engineering review and one independent methodology review;
  every blocker was fixed or reported (see the research report).

## Next executable actions (prioritised)
1. **Multi-attacker responses.** The off-ball study moves one attacker at a time; real runs are
   joint. A joint-innovation counterfactual with `learned_innovation` is the natural next study.
2. **Possession-value target.** Replace the best-progressive-option measure with a
   possession-value model so off-ball credit is measured in a quantity that sees small moves.
3. **Structured fitted-range fields.** `learned_innovation`'s fitted range is still prose that
   the lab parses; expose share, limit and horizon as fields.
4. **More Metrica-like data.** Three optical matches limit every Metrica CI to within-match
   sampling.

## Decisions needing the owner
* Whether to host a public demo (Metrica has no licence file; SkillCorner is MIT with a
  credit request, and the app shows SkillCorner team names, competition and date).
* Whether to open a PR from this branch to `main`.
* Whether to change the managed stop hook or enable commit signing.

## Resume
```bash
git checkout claude/loving-wright-i0shmp && uv sync && (cd web && npm ci)
uv run footymind download && uv run footymind ingest && uv run footymind build-similarity-index
scripts/check.sh && (cd web && npx playwright test --trace off)
scripts/reproduce.sh --dry-run
```
