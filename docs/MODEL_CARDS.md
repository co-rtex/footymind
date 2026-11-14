# Model cards

These are the models the FootyMind application serves. Research-only models (the pass-outcome
families `A/B/C`, `Adt/Bdt`, and the temporal variants) are documented in
[`RESEARCH_REPORT.md`](RESEARCH_REPORT.md) and are not served. Numbers cite
[`CLAIMS_LEDGER.md`](CLAIMS_LEDGER.md).

---

## 1. Pass-option model (one per provider)

| | |
|---|---|
| Files | `src/footymind/spatial/option_model_v2.json` (Metrica), `option_model_v2_skillcorner.json` (SkillCorner). Plain-JSON logistic coefficients plus 32 possession-clustered bootstrap members; no pickles |
| Output | For each teammate of the ball carrier, an estimated probability that a pass to them would be completed. Provenance: **estimated** |
| Inputs | Decision-frame positions and causal velocities. For each target: distance, angle, forward progress, pitch control at the target and at a lead point, lane clearance and margin, the nearest marker, target speed along the pass. No event end locations, receivers or outcomes |
| Training data | Attempted passes (rules v2). Metrica: 3 matches. SkillCorner: 20 matches, trained only on SkillCorner |
| Held-out performance (LOMO) | Metrica: log loss 0.414, AUC 0.796, ECE 0.019 (C21). SkillCorner: 0.354 / 0.811 / 0.024 (C24) |
| Uncertainty | Bootstrap members drive scenario sign agreement (how many members agree on the direction of a change) |
| Intended use | Comparing passing options within a frame, and comparing observed vs edited frames in the Multiverse |
| Not for | Valuing players. Predicting outcomes of passes that were never attempted (estimates for options nobody chose are extrapolations). Using one provider's model on another provider's data (the API refuses to) |
| Known limits | Few Metrica matches. Broadcast extrapolated positions enter as estimates. Labels depend on per-provider event mapping |

## 2. Physics geometry (pitch control and lanes)

| | |
|---|---|
| Files | `src/footymind/spatial/pitch_control.py`, `lanes.py` |
| Output | A pitch-control surface (Spearman 2018 time-to-intercept model) and lane margins and interception heuristics. Provenance: **estimated / derived** |
| Status | Used for **geometry and visual explanation only**. As probabilities it is miscalibrated (ECE 0.18), and fitted parameters reach implausible bounds (C7, C15) |

## 3. Opponent response policies

| policy | what it does | evidence | served status |
|---|---|---|---|
| `none` | no response | — | no response simulated |
| `marker_follows` | the nearest marker shifts with a moved attacker | passes the observational innovation check on both providers, β ≈ 1.07 / 1.30 (C64) | experimental (consistent with real reactions) |
| `lane_cover` | the best-placed defender moves toward the new passing lane | passes on both providers, β ≈ 0.77 / 0.80 (C64) | experimental (consistent with real reactions) |
| `press_carrier` | the nearest defender closes on a moved carrier | not supported under envelope v2 (C64) | experimental |
| `zonal_shift` | the block shifts with the ball | overshoots real movement, β 0.13–0.25 (C64) | experimental, illustration only |
| `learned` | ridge forecast of short-horizon defender displacement | beats constant velocity at forecasting real play (C40, C42); nearly inert under edits (C44, C61) | validated **as a forecaster only** |
| `learned_innovation` | ridge model of defenders' deviation from their own motion given attackers' unexpected movement | beats constant velocity and a permuted placebo on both providers (C62); moves defenders about 1.3–1.5 m under a 5 m edit (C63) | validated (observational association, not causal) |

All moves are limited by the kinematic envelope v2 (direction-dependent acceleration,
9 m/s top speed, lag and noise margins; C55–C58) and labelled **simulated**. No policy has been validated as a response to an intervention that
never happened. There is no ground truth for that, and the served assumptions say so.

## 4. Similar-moment retrieval

| | |
|---|---|
| Files | `src/footymind/research/similarity.py`. The index lives at `data/processed/similarity_<provider>.parquet` (derived, not committed) |
| Output | The most similar pass decisions from **other matches of the same provider** (with observed outcomes and deep links), plus the validated estimate: the completion rate among the 50 nearest, shrunk toward the base rate |
| Features | Standardised decision-time spatial context, time-on-ball and 1 s movement history, read 0.2 s before the release (ADR-0007) |
| Held-out performance | The k = 50 shrunk estimate beats the base rate on both providers (Δ log loss −0.057 / −0.051) and matches the logistic model on SkillCorner (C36) |
| Not for | Predicting the current pass. The listed neighbours are examples to inspect, and their own share is not the validated estimate |

## 5. Evidence-grounded explanations

| | |
|---|---|
| Files | `src/footymind/spatial/explain.py` |
| Type | Deterministic templates. No language model |
| Guarantees | Each statement names the metric it comes from, its value, its provenance (estimated when any input position is an extrapolation) and the players it mentions. No metric means no statement |
| Limits | Thresholds such as "under pressure" are fixed descriptive cut-offs, not fitted |
