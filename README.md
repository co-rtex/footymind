# FootyMind: The Invisible Game

**A tracking-based football analysis lab for the part of the game that statistics
miss: space, passing options, defensive influence, and "what if?" scenarios grounded in
real player positions.**

FootyMind reconstructs real possessions from synchronised player-tracking and event data
and quantifies passing options and defensive pressure with a physics-based model. It also
lets you *move a player* to see how the model's picture of the play changes. The
hypothetical is checked against physical reachability, shown side by side with the
observed state, and reported with its sensitivity to the model's assumptions.

> Status (2026-11-13): backend, analytics, counterfactual engine and the first research
> study are implemented and tested. The interactive frontend is in active development.
> See [`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md).

## What is in here

| Capability | What it does | Status |
|---|---|---|
| Real tracking ingestion | Metrica Sports sample games: 25 Hz positions for 22 players + ball, synchronised events. Checksum-pinned downloads, data-quality gates, versioned bundles | ✅ |
| Spatial analytics | Pitch control (Spearman 2018 time-to-intercept model), passing-lane geometry and interception heuristic, pressure on the ball carrier | ✅ (uncalibrated estimates, labelled as such) |
| Possession Multiverse | Relocate players at a real frame → reachability check → recompute options → compare with observed → parameter-sensitivity ensemble → explicit assumptions | ✅ API · 🚧 UI |
| Research: does spatial context matter? | Leakage-audited attempted-pass outcome study with event-only, spatial and interaction model families, ablations and sensitivity analyses | ✅ v1 |
| Ghost Value / Defensive Shadow | Off-ball and defensive valuation | 🔬 planned (building blocks exist) |

## Key research finding (v1)

On held-out matches (leave-one-match-out, 2,236 attempted passes, Metrica Games 1–2):

* Adding tracking-derived spatial context to event-level information reduced log loss from
  **0.438 to 0.369** (Δ −0.069, 95% CI [−0.089, −0.050]) and raised AUC from **0.78 to 0.86**.
* Most of that gain comes from **teammate geometry relative to the pass direction**.
  **Defender-only** information adds a smaller but detectable improvement (Δ −0.015 [−0.025, −0.005]).
* **Negative results:** richer interaction features did not help beyond the spatial model.
  The literature-parameter physics estimate is **poorly calibrated** on observed passes
  (ECE 0.18). This is why every model estimate in the product is labelled "uncalibrated".

These results replicate established findings (Power et al. 2017; Anzer & Bauer 2022) on
open data, with an explicit leakage audit. They are not a new discovery. Two matches from
one provider cannot support generalisation claims. See [`docs/RESEARCH.md`](docs/RESEARCH.md)
and the [claims ledger](docs/CLAIMS_LEDGER.md).

## Scientific guardrails

* **Provenance on everything.** Each value is *observed*, *derived*, *estimated*,
  *hypothetical* or *simulated*, and the API and UI keep these distinct.
* **No fabricated observations.** Unobserved positions are `null`. Tracking artefacts are
  flagged, not smoothed away.
* **No leakage.** Decision-time features use causal velocities only. Failed-pass end
  locations, which are the *loss location* in Metrica, are excluded.
* **Counterfactuals are not causal claims.** Scenarios are model-based comparisons with
  feasibility checks, stated assumptions and a parameter ensemble showing whether the
  direction of change is robust.

## Quick start

```bash
uv sync
uv run footymind download      # Metrica sample games, SHA-256 verified (~180 MB)
uv run footymind ingest        # build processed bundles (~3 s per match)
uv run pytest -q               # unit + real-data tests
uv run footymind serve         # API at http://127.0.0.1:8000 (docs at /docs)
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
(https://github.com/metrica-sports/sample-data), which is anonymised. Raw data is
downloaded at setup and is not redistributed in this repository. See
[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## History

FootyMind v1, a match-outcome predictor trained on a small synthetic sample, is archived
with an audit of its limitations in [`legacy/match_outcome_predictor`](legacy/match_outcome_predictor).
