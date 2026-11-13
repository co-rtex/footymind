# FootyMind: The Invisible Game

**A tracking-based football analysis lab for the part of the game that statistics
miss: space, passing options, defensive influence, and "what if?" scenarios grounded in
real player positions.**

FootyMind reconstructs real possessions from synchronised player-tracking and event data
and quantifies passing options and defensive pressure with a physics-based model. It also
lets you *move a player* to see how the model's picture of the play changes. The
hypothetical is checked against physical reachability, shown side by side with the
observed state, and reported with its sensitivity to the model's assumptions.

> Status (2026-11-13): end-to-end flagship flow working on real data. That covers the replay,
> pause-and-analyse, dragging a player, comparing with uncertainty, and resetting. See
> [`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md).

![Frame analysis: pitch control, ranked pass options with calibrated and physics estimates](web/screenshots/lab-analysis-desktop.png)

![Possession Multiverse: a moved player (hypothetical), the marker's rule-based response (simulated), and the pitch-control difference](web/screenshots/scenario-difference-desktop.png)

## What is in here

| Capability | What it does | Status |
|---|---|---|
| Real tracking ingestion | Metrica Sports sample games 1–3 (CSV and EPTS formats): 25 Hz positions for 22 players + ball, synchronised events. Checksum-pinned downloads, data-quality gates, versioned bundles | ✅ |
| Spatial analytics | Pitch control (Spearman 2018 time-to-intercept model), passing-lane geometry and interception heuristic, pressure on the ball carrier | ✅ (uncalibrated estimates, labelled as such) |
| Possession Multiverse | Relocate players at a real frame → reachability check → recompute options → compare with observed → parameter-sensitivity ensemble → explicit assumptions | ✅ |
| Calibrated option model | Logistic model over per-option features, evaluated leave-one-match-out (log loss 0.422, AUC 0.789, ECE 0.017), served as plain-JSON coefficients | ✅ |
| Interactive lab (web) | Canvas replay at 60 fps, overlays, drag-to-edit scenarios, observed/hypothetical/difference views, evidence panel | ✅ |
| Research: does spatial context matter? | Leakage-audited attempted-pass outcome study with event-only, spatial and interaction model families, ablations and sensitivity analyses | ✅ v1 |
| Ghost Value: Defensive Shadow | Per-defender suppression of passing value against reachable alternative positions | 🧪 experimental (API only, not yet validated) |

## Key research finding (v1)

On held-out matches (leave-one-match-out, 3,633 attempted passes, Metrica Games 1–3):

* Adding tracking-derived spatial context to event-level information reduced log loss from
  **0.429 to 0.358** (Δ −0.071, 95% CI [−0.085, −0.057]) and raised AUC from **0.78 to 0.87**.
  The gain holds in every held-out match, including one with a different annotation scheme.
* Most of that gain comes from **teammate geometry relative to the pass direction**.
  **Defender-only** information adds a smaller but consistent improvement (Δ −0.014 [−0.021, −0.008]).
* **Negative results:** richer interaction features did not help beyond the spatial model.
  The literature-parameter physics estimate is **poorly calibrated** on observed passes
  (ECE 0.18). This is why every model estimate in the product is labelled "uncalibrated".

These results replicate established findings (Power et al. 2017; Anzer & Bauer 2022) on
open data, with an explicit leakage audit. They are not a new discovery. Three matches from
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

Run the web lab (in a second terminal, with the API running):

```bash
cd web && npm ci && npm run dev    # http://localhost:5173 (proxies /api to :8000)
```

Or build it once (`cd web && npm run build`) and `footymind serve` will serve it at `/`.
Backend and frontend checks are listed in [`CLAUDE.md`](CLAUDE.md) and [`web/README.md`](web/README.md).

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
