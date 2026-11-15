# Current state and session handoff

_Last updated: 2026-11-15, production audit of the released v1.0.0 and its public demo.
Keep this file factual and update it at every checkpoint._

## Release and deployment (2026-11-15)
* **v1.0.0 is released** (tag `v1.0.0` at 22572ab, the squash-merged release; GitHub release
  published by `.github/workflows/publish-release.yml`). `main` also has ffa0aff (Railway
  Dockerfile builder) and 6d4693a (release docs); CI run #89 on 6d4693a passed.
* **Public demo:** https://footymind-web-production.up.railway.app/ (Railway, SkillCorner-only:
  matches 1874553, 1899585, 2017461; settings in `docs/release/DEPLOYMENT.md`).
* **Production audit (2026-11-15).** Verified against the live site:
  * `smoke_api.py --require-similar` passed.
  * 117 of 141 production API responses are identical to a local server on the same code and
    bundles: summaries, possessions, replays, analyses, Defensive Shadow, and scenarios with a
    fixed seed. The 24 that differ are similar-moments neighbours, because the demo index has
    3 matches against 20 locally.
  * Browser tests passed: 3 viewports × 3 matches, the guided tour, all 7 response policies,
    similar moments and error routes, with no runtime errors, failed requests or horizontal
    overflow.
  * Findings fixed on this branch (PR co-rtex/footymind#3, not yet deployed):
    * the demo's similar-moments panel claimed a validation done on 20 matches;
    * no rate limiting on the free-tier instance;
    * the server ran as root on Railway;
    * missing security headers, `HEAD /api/health` returned 404, unknown paths returned raw
      JSON 404s;
    * relative links in the release notes.
  * Owner actions:
    * correct the GitHub repository description, which still describes the archived v1
      predictor;
    * add the demo URL as the website, and add topics;
    * optionally fix the release body's relative links;
    * set `FOOTYMIND_CLIENT_IP_HEADER=X-Real-IP` on Railway after merging.

## Branch and repository
* Working branch: `claude/loving-wright-i0shmp`, restarted from `main` after the v1.0.0 squash
  merge (an `ours` merge keeps the pre-release history without a force push). Audit fixes are
  in PR co-rtex/footymind#3.
* Commits are authored as Christian Cortez with no assistant trailers (see CLAUDE.md).
* All specialist worktrees and branches were merged and removed; no other worktrees exist.
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
uv run footymind ingest all        # all 23 sources (needs `footymind download all`, ~2 GB)
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
  `no_response`). `learned_innovation` also serves its per-provider `fitted_range` (99% of real
  1 s attacker innovations below 2.29 m Metrica / 2.48 m SkillCorner, read from
  `response_diagnostics_v1*`). Kinematic **envelope v2** clips every simulated move.
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
  request-body cap (413), cache freshness after re-ingest, stale or unreadable bundles left out
  of `/api/matches` (listed as `unavailable` in `/api/health`) with 503 on their own endpoints.
* Docs and ledger consistency are tested (`tests/test_claims_ledger.py`).
* Docker image verified at runtime in CI on every push: first start on an empty volume
  (Metrica 1–2 downloaded by the entrypoint, `smoke_api.py --require-similar`, `HEALTHCHECK`),
  then `scripts/container_recovery.py docker` (healthy restart, outdated bundle, interrupted
  first start, unknown source and damaged file, restart on the damaged volume).
* The CLI names its sources (`metrica`, `skillcorner`, `all` or ids); a bare `download` stops.
* Release documents: `docs/release/` (release notes, checklist, verification, deployment,
  data licences), `docs/ROADMAP.md` (post-1.0 ideas), `docs/product/release-1.0/QA_REPORT.md`.

## Verification (final release session)
| Gate | Result |
|---|---|
| `scripts/check.sh` (exit 0; all 23 bundles at pipeline v3) | ruff, format and mypy on `src`, `tests`, `scripts`; **355 pytest passed, 0 skipped**; web typecheck, lint, **222 vitest**, build |
| `cd web && npx playwright test --trace off` | **31/31** against the real API on all 23 matches; **25/25** on Metrica-only data (`--grep-invert @skillcorner`, as CI runs it) |
| Entrypoint recovery | `scripts/container_recovery.py local`: 49/49 checks; `docker` mode in CI (see `docs/release/RELEASE_CHECKLIST.md`) |
| Clean environment | fresh clone + empty data dir, README minimal path: pytest 317 passed / 14 skipped (no SkillCorner), build, smoke, 8/8 Metrica flow tests |
| Experiment reproduction | `pass_outcome_v3` (C20) re-run at 259c0be: identical to the committed result |
| Security | hostile-input sweep (no 500, no leaks); `pip-audit` clean; `npm audit --omit=dev` 0 |
| GitHub CI | the run for the final commit is recorded in `docs/release/RELEASE_CHECKLIST.md` |

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
* Independent reviews: session 4 ran an engineering review, a methodology review, a numbers
  audit and a code review. The final session ran a research-claims audit (4 medium, 11 low
  findings, all fixed in 22070db), a product QA in the browser (8 defects fixed, 7 low/info
  documented in `docs/product/release-1.0/QA_REPORT.md`) and a final release-blocker review
  (2 medium, 7 low, all fixed in ae9188c; no critical or high findings).

## After 1.0
FootyMind 1.0 is feature-frozen. Post-1.0 research and product ideas are in
[`docs/ROADMAP.md`](ROADMAP.md); none of them is unfinished 1.0 work.

## Decisions needing the owner
* Merge PR co-rtex/footymind#3, which redeploys the demo from `main`, and set
  `FOOTYMIND_CLIENT_IP_HEADER=X-Real-IP` on Railway.
* Repository settings: the description (still the archived v1 predictor's), the website (the
  demo URL) and topics.
* Keep the public demo SkillCorner-only unless Metrica's terms for hosted use are confirmed
  (`docs/release/DATA_LICENSES.md`). The code is MIT-licensed (`LICENSE`).
* Whether to change the managed stop hook or enable commit signing.

## Resume
```bash
git checkout claude/loving-wright-i0shmp && uv sync && (cd web && npm ci)
uv run footymind download metrica && uv run footymind ingest metrica   # or `all` (~2 GB)
uv run footymind build-similarity-index
scripts/check.sh && (cd web && npx playwright test --trace off)
scripts/reproduce.sh --dry-run
```
