# FootyMind: The Invisible Game

**A tracking-based football analysis lab for the part of the game that statistics
miss: space, passing options, defensive influence, and "what if?" scenarios grounded in
real player positions.**

FootyMind reconstructs real possessions from synchronised player-tracking and event data
and quantifies passing options and defensive pressure with a physics-based model. It also
lets you *move a player* to see how the model's picture of the play changes. The
hypothetical is checked against physical reachability, shown side by side with the
observed state, and reported with its sensitivity to the model's assumptions.

> Status (2026-11-13): end-to-end flagship flow working on real data from two providers (23
> matches). That covers the replay, pause-and-analyse, dragging a player, comparing with
> uncertainty, and resetting. The main research result is replicated across providers. See
> [`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md).

![Frame analysis: pitch control, ranked pass options with calibrated and physics estimates](web/screenshots/lab-analysis-desktop.png)

![Possession Multiverse: a moved player (hypothetical), the marker's rule-based response (simulated), and the pitch-control difference](web/screenshots/scenario-difference-desktop.png)

![Broadcast tracking (SkillCorner): off-camera players are provider extrapolations, drawn hollow and labelled as estimates](web/screenshots/broadcast-estimated-desktop.png)

## What is in here

| Capability | What it does | Status |
|---|---|---|
| Real tracking ingestion | Metrica Sports sample games 1–3 (CSV and EPTS) and 20 SkillCorner A-League matches (broadcast, off-camera extrapolations labelled as estimates): player and ball positions (25 Hz / 10 Hz) with synchronised events. Checksum-pinned downloads, data-quality gates, versioned bundles | ✅ |
| Spatial analytics | Pitch control (Spearman 2018 time-to-intercept model), passing-lane geometry and interception heuristic, pressure on the ball carrier | ✅ (uncalibrated estimates, labelled as such) |
| Possession Multiverse | Relocate players at a real frame → reachability check → recompute options → compare with observed → parameter-sensitivity ensemble → explicit assumptions | ✅ |
| Calibrated option model | Logistic model over per-option features incl. lead-pass targets, evaluated leave-one-match-out (v2: log loss 0.414, AUC 0.796, ECE 0.019), served as plain-JSON coefficients with bootstrap uncertainty | ✅ |
| Interactive lab (web) | Canvas replay at 60 fps, overlays, drag-to-edit scenarios, observed/hypothetical/difference views, evidence panel | ✅ |
| Research: does spatial context matter? | Leakage-audited attempted-pass outcome study with event-only, spatial and interaction model families, ablations and sensitivity analyses | ✅ v1 |
| Ghost Value: Defensive Shadow | Per-defender suppression of passing value against nearby alternative positions | 🧪 experimental (API + UI layer; not better than nearest-defender, at chance on SkillCorner) |
| Opponent response policies | Defender reactions to a scenario edit (`press_carrier`, `zonal_shift`, `lane_cover`, `marker_follows`, `learned`) limited by a kinematic envelope (reaction delay, acceleration, top speed), with a static-vs-response comparison in the lab | ✅ `learned` validated as a short-horizon forecast of real play; rule policies 🧪 experimental |
| Temporal context | Causal time-on-ball and last-second movement history at a guard frame before the release (ADR-0007), served with the frame analysis | ✅ research study RQ2 (see findings) |
| Why this moment | Deterministic, evidence-grounded statements: each sentence is tied to one computed metric, its provenance and the players it mentions | ✅ |
| Similar moments | The most similar pass decisions from *other* matches of the same provider, with observed outcomes and deep links | ✅ validated held-out (beats the base rate on both providers) |

## Key research findings

On held-out matches (leave-one-match-out), two questions are kept separate after an
independent methodology review. The primary result is replicated across two providers:
Metrica (3 matches, optical, 25 Hz) and SkillCorner (20 A-League matches, broadcast, 10 Hz).

* **Before the pass is played** (decision-time, nothing from the kick): adding tracking-derived
  spatial context reduces log loss by **Δ −0.026 [−0.034, −0.018]** on Metrica (AUC 0.65 → 0.72)
  and by **Δ −0.021 [−0.025, −0.018]** on SkillCorner (15,914 passes). The gain holds in every
  held-out match of both providers (3/3 and 20/20).
* **Given the executed kick** (action-conditioned: release direction and speed known):
  Δ −0.071 on Metrica and −0.046 on SkillCorner. This is provider-dependent, because much of it
  is where the ball was actually played relative to teammates, which is partly execution quality.
* **Defender-only** information adds a small but consistent improvement (Δ −0.015 / −0.018).
  Interaction features add nothing on 3 matches and a small, significant amount (−0.004) on 20.
* **Negative results:** the literature physics model is poorly calibrated (ECE 0.17), and
  fitting its parameters drives them to implausible bounds, so it is structurally
  misspecified. The experimental *Defensive Shadow* metric is robust to tracking noise and
  finds interceptors at 2× chance on Metrica, but it is **not** better than "nearest defender", and
  on SkillCorner it is at chance (the 2× signal does not replicate).
* **The last second matters (session 3, RQ2).** Adding 1 s of movement history (pressure
  building, ball travel, team and defensive-line movement) to decision-time spatial context
  improves held-out prediction on both providers: **Δ −0.014** on Metrica and **Δ −0.009** on
  SkillCorner, better in 20/20 matches. Every feature is read **0.2 s before the release**,
  because on broadcast tracking the event frame often already contains the kick (ADR-0007).
  The headline spatial result above is unchanged under that guard (C33–C35, C38).
* **Off-ball credit is convention-dependent.** In the second before a pass, attacking movement
  raises the best progressive option and defensive movement takes about as much back, on
  average. The option model is so non-additive that per-pass credit for a run depends on
  which side is held fixed. A placebo shows this is mostly the model, with a small
  genuine attacker–defender coupling on both providers (C31, C32).
* **Opponent responses.** A learned position model forecasts real defender movement better than
  constant velocity but is inert under edits. Positions barely predict how defenders deviate
  from their own motion (C61). A model of *co-movement* with attackers' unexpected movement
  beats constant velocity and a placebo on both providers and is served as an observational
  response (C62). Of the interpretable rules, `marker_follows` and `lane_cover` match real
  reactions on both providers; `press_carrier` and `zonal_shift` do not (C64).
* **Similar moments are informative.** The completion rate among the 50 most similar decisions
  from other matches beats the base rate held-out on both providers (−0.057 / −0.051 log loss)
  and matches the logistic model on SkillCorner (C36).

![Cross-provider replication of the main effects](docs/figures/replication_v1.png)

![RQ2: temporal context at the release frame and at the 0.2 s guard frame, per provider](docs/figures/rq2_temporal_v1.png)

These results are consistent with established findings (Power et al. 2017; Anzer & Bauer 2022)
on open data, with an explicit leakage audit and a cross-provider replication. They are not a
new discovery, and two leagues' worth of samples cannot support universal claims. See [`docs/RESEARCH.md`](docs/RESEARCH.md)
and the [claims ledger](docs/CLAIMS_LEDGER.md).

## Scientific guardrails

* **Provenance on everything.** Each value is *observed*, *derived*, *estimated*,
  *hypothetical* or *simulated*, and the API and UI keep these distinct.
* **No fabricated observations.** Unobserved positions are `null`. Tracking artefacts are
  flagged, not smoothed away. Broadcast positions that the provider extrapolated off camera are
  labelled *estimated* and drawn hollow, never presented as observed.
* **Providers are never pooled.** Metrica and SkillCorner differ in frame rate, visibility and
  labels, so each is evaluated separately and has its own option model.
* **No leakage.** Decision-time features use causal velocities only. Failed-pass end
  locations, which are the *loss location* in Metrica, are excluded.
* **Counterfactuals are not causal claims.** Scenarios are model-based comparisons with
  feasibility checks, stated assumptions and a parameter ensemble showing whether the
  direction of change is robust.

## Quick start

```bash
uv sync
# Metrica sample games, SHA-256 verified (~180 MB), then processed bundles (~3 s per match)
uv run footymind download metrica-sample-1 metrica-sample-2 metrica-sample-3
uv run footymind ingest metrica-sample-1 metrica-sample-2 metrica-sample-3
uv run footymind build-similarity-index --provider metrica   # "Similar moments" index (~40 s)
uv run pytest -q             # unit + real-data tests
uv run footymind serve       # API at http://127.0.0.1:8000 (docs at /docs)
```

Without match ids, `download` and `ingest` take **every** registered source, including the
20 SkillCorner matches (about 1.8 GB more). Data lives in `./data` unless `FOOTYMIND_DATA_DIR`
points elsewhere. Without the similarity index the API answers `/similar` with a 503 and the
lab shows a "no index" note.

Run the web lab (in a second terminal, with the API running):

```bash
cd web && npm ci && npm run dev    # http://localhost:5173 (proxies /api to :8000)
```

Or build it once (`cd web && npm run build`) and `footymind serve` will serve it at `/`.
Backend and frontend checks are listed in [`CLAUDE.md`](CLAUDE.md) and [`web/README.md`](web/README.md).

Or run everything in one container (the first start downloads and ingests the three Metrica
games and builds the similarity index, about a minute; set `FOOTYMIND_SOURCES` to choose
other sources):

```bash
docker build -t footymind .
docker run --rm -p 8000:8000 -v footymind-data:/data footymind   # http://localhost:8000
scripts/container_smoke.sh   # build, run on a copy of local data and check it end to end
```

Reproduce the research results:

```bash
uv run footymind experiment experiments/configs/pass_outcome_v1.json
uv run footymind experiment experiments/configs/pass_outcome_v1_ablation.json
```

Try a counterfactual through the API:

```bash
curl -s -X POST http://127.0.0.1:8000/api/scenarios -H 'content-type: application/json' \
  -d '{"match_id":"metrica-sample-1","frame":20666,"interventions":[{"player_id":"home_2","x":-3.0,"y":-20.0}]}'
```

## Architecture

A modular monolith: a Python package (`footymind`) with provider adapters → canonical
tracking model → spatial models → counterfactual engine and research pipeline, served by
FastAPI. The frontend is Vite + React + TypeScript with a Canvas pitch renderer. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and the [ADRs](docs/adr/).

## Data and attribution

Tracking and event data: **Metrica Sports sample data**
(https://github.com/metrica-sports/sample-data), which is anonymised, and **SkillCorner open
data** (https://github.com/SkillCorner/opendata, MIT; credit SkillCorner). Raw data is
downloaded at setup and is not redistributed in this repository. See
[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## History

FootyMind v1, a match-outcome predictor trained on a small synthetic sample, is archived
with an audit of its limitations in [`legacy/match_outcome_predictor`](legacy/match_outcome_predictor).
