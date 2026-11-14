# Experiment `temporal_guard_v1_04s`

- Code revision: `379b98eea585e3a4754166ac9bbc27bdc9e35a95`
- Matches: metrica-sample-1, metrica-sample-2, metrica-sample-3 (leave-one-match-out)
- Attempted passes: 3592 (completion rate 0.796); label sources {'BALL LOST': 598, 'BALL OUT': 136, 'PASS': 2858}
- Receiver inference (completed passes): 0.740 top-1 (n=2858)
- Runtime: 1673.2 s

## Pooled out-of-fold metrics

| model | family | log loss | Brier | AUC | ECE |
|---|---|---|---|---|---|
| Bdt_tt_gbm | temporal | 0.4414 | 0.1412 | 0.750 | 0.029 |
| Bdt_tt_logreg | temporal | 0.4465 | 0.1418 | 0.738 | 0.025 |
| Bdt_tob_gbm | temporal | 0.4522 | 0.1450 | 0.733 | 0.020 |
| Bdt_tob_logreg | temporal | 0.4549 | 0.1451 | 0.719 | 0.020 |
| Bdt_logreg | B-decision | 0.4552 | 0.1453 | 0.720 | 0.021 |
| Adt_tob_logreg | temporal | 0.4805 | 0.1531 | 0.648 | 0.013 |
| Adt_logreg | A-decision | 0.4820 | 0.1536 | 0.644 | 0.016 |
| base_rate | baseline | 0.5081 | 0.1631 | 0.462 | 0.028 |

## Per-fold log loss (test match)

| model | metrica-sample-1 | metrica-sample-2 | metrica-sample-3 |
|---|---|---|---|
| base_rate | 0.5554 | 0.4750 | 0.5009 |
| Adt_logreg | 0.5343 | 0.4400 | 0.4786 |
| Bdt_logreg | 0.5071 | 0.4164 | 0.4494 |
| Adt_tob_logreg | 0.5324 | 0.4376 | 0.4782 |
| Bdt_tob_logreg | 0.5057 | 0.4162 | 0.4497 |
| Bdt_tt_logreg | 0.4892 | 0.4060 | 0.4489 |
| Bdt_tob_gbm | 0.5001 | 0.4163 | 0.4469 |
| Bdt_tt_gbm | 0.4785 | 0.4108 | 0.4397 |

## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)

| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|
| Adt_logreg | Bdt_logreg | -0.0268 [-0.0354, -0.0188] | -0.0083 [-0.0114, -0.0054] | +0.0757 [+0.0574, +0.0950] |
| Adt_logreg | Adt_tob_logreg | -0.0015 [-0.0040, +0.0010] | -0.0005 [-0.0013, +0.0003] | +0.0046 [-0.0039, +0.0132] |
| Bdt_tob_logreg | Bdt_tt_logreg | -0.0084 [-0.0145, -0.0023] | -0.0033 [-0.0056, -0.0011] | +0.0185 [+0.0090, +0.0282] |
| Bdt_tob_gbm | Bdt_tt_gbm | -0.0108 [-0.0168, -0.0046] | -0.0038 [-0.0061, -0.0015] | +0.0171 [+0.0077, +0.0264] |
| Adt_tob_logreg | Bdt_tob_logreg | -0.0256 [-0.0340, -0.0176] | -0.0080 [-0.0111, -0.0052] | +0.0710 [+0.0529, +0.0897] |

## Held-out permutation importance: Bdt_tt_logreg (Δ log loss, top 10)

- dist_to_goal: +0.1515
- ball_x_att: +0.0248
- passer_fwd_m_l10: +0.0238
- att_width_m: +0.0157
- ball_travel_m_l10: +0.0156
- prev_type_pass: +0.0144
- def_goal_side_of_ball: +0.0136
- att_depth_m: +0.0111
- def_centroid_fwd_m_l10: +0.0105
- press_closing_speed: +0.0094

## Caveats

- Only these matches are used, all from one anonymised provider sample. CIs reflect within-match sampling of possessions, not between-match or between-competition variation.
- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).
- Failed-pass labels depend on the documented Metrica event-mapping rules.
