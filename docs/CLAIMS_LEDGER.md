# Claims ledger

The only source of quotable numbers. Each claim names its evidence, data, code revision,
procedure and limitations. A claim that cannot be traced here must not appear in the
README, a report or a CV.

Data: Metrica Sports sample data, Games 1–2, with files pinned by SHA-256 in
`src/footymind/data/sources.py`. Evidence revision for experiment claims: the
`code_revision` field in each `experiments/results/<name>/results.json`.

| # | Claim | Evidence | Procedure | Limitations | Status |
|---|---|---|---|---|---|
| C1 | Event/tracking synchronisation: the median passer–ball distance at pass start frames is 0.084 m (G1) and 0.088 m (G2); the receiver–ball distance at pass end is 0.22 m | ingestion manifest `quality.event_sync`; `tests/test_real_metrica.py::test_event_tracking_synchronisation` | All PASS events with a known passer | Metrica only | confirmed |
| C2 | Attacking direction inferred from goalkeeper positions is consistent with every goal location | `test_goals_scored_towards_attacking_goal` | Goal shots at x·sign > 30 m | 8 goals in total | confirmed |
| C3 | Spatial features reduce held-out log loss for attempted-pass outcome from 0.438 to 0.369 (logistic regression); Δ −0.069 [95% CI −0.089, −0.050] | `experiments/results/pass_outcome_v1/results.json` | LOMO, 2,236 passes, possession-clustered paired bootstrap (2,000) | 2 matches, one provider; within-match CI only | confirmed |
| C4 | AUC improves from 0.777 (event-only) to 0.863 (spatial), logistic regression | same | same | same | confirmed |
| C5 | Defender-only information (no teammate positions) gives Δ log loss −0.015 [−0.025, −0.005] over event-only | `pass_outcome_v1_ablation` | LR ablation, same protocol | Small effect. Positive in both sensitivity analyses | confirmed |
| C6 | Interaction features (Model C) do not significantly improve on spatial features (Model B): GBM Δ −0.007 [−0.016, +0.003] | `pass_outcome_v1` | same | Small data | confirmed (negative result) |
| C7 | The literature-parameter PPCF × lane estimate is poorly calibrated on observed passes (ECE 0.184), and after Platt recalibration it is worse than the event baseline (Δ log loss +0.026 [+0.009, +0.043]) | `pass_outcome_v1` | raw estimate for the inferred receiver | Depends on receiver inference and the static-target assumption | confirmed (negative result) |
| C8 | Release-direction receiver inference matches the true receiver on 78.6% of completed passes | `pass_outcome_v1` dataset summary | n = 1,763 completed passes | Completed passes only (true target unknown for failures) | confirmed |
| C9 | The C3/C5 effects persist when excluding BALL OUT labels (A→B −0.056 [−0.076, −0.038]) and truncated release windows (−0.062 [−0.083, −0.043]) | `pass_outcome_v1_sens_*` | same protocol | same | confirmed |
| C10 | Frame analysis latency is about 70 ms warm; a scenario with a 24-member ensemble takes about 0.3 s; possession replay takes 29 ms | manual timing on the development container (4 vCPU) | single measurements | Not a benchmark suite; hardware-dependent | indicative only |

## Not claimed
* No causal effect of any intervention is claimed. Scenarios are model-based comparisons.
* No claim about generalisation to other competitions, providers or broadcast tracking.
* No player names or team identities (the data is anonymised).
