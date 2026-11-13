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
| C11 | With three matches (3,633 passes), spatial features reduce held-out log loss from 0.429 to 0.358 (LR; Δ −0.071 [−0.085, −0.057]); the improvement holds in each of the 3 test folds, including the Game 3 transfer fold (0.422 → 0.364) | `experiments/results/pass_outcome_v2_3match/results.json` | LOMO (3 folds), same protocol | Game 3 uses a different annotation scheme (CARRY events); one provider | confirmed |
| C12 | Defender-only information: Δ log loss −0.014 [−0.021, −0.008] (3 matches); C vs B GBM −0.002 [−0.006, +0.003] (no gain) | same | same | same | confirmed |
| C13 | Game 3 EPTS ingestion: the inferred attacking direction matches the provider-declared direction, and the passer–ball sync median is 0.065 m | ingestion manifest; `metrica_epts.load_tracking` raises on contradiction | — | — | confirmed |
| C14 | Option model v1 (serving): held-out (LOMO, 3 matches) log loss 0.422, AUC 0.789, ECE 0.017; v0 (15 features) was 0.428 / 0.780 / 0.019 | `src/footymind/spatial/option_model_v1.json` (meta.lomo_pooled, code_revision 0d32b0a) | `footymind train-option-model` | Trained on inferred receivers; extrapolates to unchosen options (selection bias) | confirmed |
| C15 | Fitting PPCF/lane parameters per LOMO fold improves held-out log loss 0.772 → 0.487 and ECE 0.174 → 0.072 but lowers AUC 0.751 → 0.725; the fitted parameters sit at physically implausible bounds; still worse than the option model (0.422) | `experiments/results/physics_calibration_v1/results.json` | 3 folds, 800-pass fitting subsample, 160 evaluations per fold | Subsample fitting; local optimiser; bounds chosen a priori | confirmed (negative result) |
| C16 | Defensive Shadow defender rankings are stable under σ = 0.1 m position noise (mean Spearman ρ 0.95) and σ = 0.3 m (0.84); the noise floor is about 0.025; ρ with lane presence is 0.26 | `experiments/results/shadow_stability_v1/results.json` | 30 frames, 3 noise draws each | Robustness only; no outcome validation; experimental metric | confirmed (robustness only) |
| C17 | On 386 intercepted passes, the highest-shadow defender at release is the interceptor 21.5% of the time (top-3: 38.3%) vs chance 10% / 30%; the nearest-defender baseline reaches 21.8% / 50.3% (shadow is worse on top-3 by 0.119 [0.060, 0.181]) | `experiments/results/shadow_validation_v1/results.json` | release-frame rankings, bootstrap over events | One provider, 3 matches; direction-agnostic | confirmed (partly negative) |
| C18 | Spatial gain over the event baseline is largest on releases > 20° from any teammate (Δ log loss −0.186 [−0.246, −0.125], n = 598) and robust in the final third (−0.110) and under pressure (−0.085) | `experiments/results/pass_outcome_v2_3match/failure_modes.json` | OOF predictions joined with decision-time context; possession bootstrap | Context bins chosen a priori; no multiplicity correction | confirmed |
| C10 | Frame analysis latency is about 70 ms warm; a scenario with a 24-member ensemble takes about 0.3 s; possession replay takes 29 ms | manual timing on the development container (4 vCPU) | single measurements | Not a benchmark suite; hardware-dependent | indicative only |

## Not claimed
* No causal effect of any intervention is claimed. Scenarios are model-based comparisons.
* No claim about generalisation to other competitions, providers or broadcast tracking.
* No player names or team identities (the data is anonymised).
