# Experiment `pass_outcome_v3`

- Code revision: `21310847e86ffe6304bd906130c1cd2b1b33863d`
- Matches: metrica-sample-1, metrica-sample-2, metrica-sample-3 (leave-one-match-out)
- Attempted passes: 3593 (completion rate 0.796); label sources {'BALL LOST': 598, 'BALL OUT': 136, 'PASS': 2859}
- Receiver inference (completed passes): 0.762 top-1 (n=2859)
- Runtime: 194.8 s

## Pooled out-of-fold metrics

| model | family | log loss | Brier | AUC | ECE |
|---|---|---|---|---|---|
| C_gbm | C | 0.3510 | 0.1120 | 0.865 | 0.015 |
| B_gbm | B | 0.3512 | 0.1122 | 0.866 | 0.017 |
| B_logreg | B | 0.3522 | 0.1111 | 0.869 | 0.027 |
| abl_A+target_geometry_logreg | ablation | 0.3750 | 0.1166 | 0.840 | 0.016 |
| A_gbm | A | 0.4004 | 0.1291 | 0.814 | 0.018 |
| abl_A+defender_only_logreg | ablation | 0.4083 | 0.1311 | 0.809 | 0.026 |
| option_v2_logreg | option | 0.4145 | 0.1325 | 0.796 | 0.019 |
| option_logreg | option | 0.4187 | 0.1340 | 0.789 | 0.015 |
| A_logreg | A | 0.4234 | 0.1355 | 0.785 | 0.028 |
| physics_platt | physics | 0.4509 | 0.1456 | 0.751 | 0.047 |
| Bdt_gbm | B-decision | 0.4539 | 0.1456 | 0.728 | 0.020 |
| Bdt_logreg | B-decision | 0.4562 | 0.1457 | 0.721 | 0.020 |
| Adt_gbm | A-decision | 0.4762 | 0.1522 | 0.683 | 0.028 |
| Adt_logreg | A-decision | 0.4819 | 0.1533 | 0.648 | 0.014 |
| base_rate | baseline | 0.5080 | 0.1631 | 0.463 | 0.028 |
| physics_raw | physics | 0.7670 | 0.2045 | 0.753 | 0.174 |

## Per-fold log loss (test match)

| model | metrica-sample-1 | metrica-sample-2 | metrica-sample-3 |
|---|---|---|---|
| base_rate | 0.5550 | 0.4750 | 0.5009 |
| A_logreg | 0.4618 | 0.3942 | 0.4193 |
| A_gbm | 0.4319 | 0.3702 | 0.4024 |
| physics_raw | 0.8228 | 0.7375 | 0.7505 |
| physics_platt | 0.4863 | 0.4251 | 0.4463 |
| option_logreg | 0.4638 | 0.3892 | 0.4101 |
| option_v2_logreg | 0.4577 | 0.3886 | 0.4042 |
| Adt_logreg | 0.5364 | 0.4389 | 0.4777 |
| Adt_gbm | 0.5133 | 0.4452 | 0.4747 |
| Bdt_logreg | 0.5128 | 0.4145 | 0.4492 |
| Bdt_gbm | 0.5016 | 0.4182 | 0.4486 |
| B_logreg | 0.3873 | 0.3120 | 0.3599 |
| B_gbm | 0.3783 | 0.3215 | 0.3560 |
| C_gbm | 0.3783 | 0.3191 | 0.3576 |
| abl_A+defender_only_logreg | 0.4498 | 0.3698 | 0.4100 |
| abl_A+target_geometry_logreg | 0.4047 | 0.3398 | 0.3826 |

## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)

| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|
| Adt_logreg | Bdt_logreg | -0.0257 [-0.0342, -0.0177] | -0.0077 [-0.0106, -0.0048] | +0.0732 [+0.0558, +0.0919] |
| Adt_gbm | Bdt_gbm | -0.0223 [-0.0311, -0.0135] | -0.0067 [-0.0098, -0.0037] | +0.0448 [+0.0288, +0.0607] |
| A_logreg | B_logreg | -0.0712 [-0.0843, -0.0570] | -0.0244 [-0.0294, -0.0194] | +0.0837 [+0.0700, +0.0976] |
| A_gbm | B_gbm | -0.0492 [-0.0618, -0.0360] | -0.0169 [-0.0215, -0.0123] | +0.0514 [+0.0380, +0.0643] |
| B_gbm | C_gbm | -0.0002 [-0.0045, +0.0042] | -0.0002 [-0.0018, +0.0014] | -0.0003 [-0.0041, +0.0035] |
| A_logreg | abl_A+defender_only_logreg | -0.0150 [-0.0215, -0.0086] | -0.0043 [-0.0068, -0.0018] | +0.0238 [+0.0149, +0.0323] |
| A_logreg | abl_A+target_geometry_logreg | -0.0483 [-0.0598, -0.0366] | -0.0188 [-0.0232, -0.0144] | +0.0549 [+0.0423, +0.0668] |
| option_logreg | option_v2_logreg | -0.0042 [-0.0076, -0.0007] | -0.0015 [-0.0028, -0.0001] | +0.0062 [+0.0016, +0.0104] |
| Adt_logreg | A_logreg | -0.0586 [-0.0695, -0.0468] | -0.0178 [-0.0220, -0.0135] | +0.1377 [+0.1160, +0.1595] |

## Held-out permutation importance: Bdt_logreg (Δ log loss, top 10)

- dist_to_goal: +0.1167
- prev_type_pass: +0.0268
- ball_x_att: +0.0224
- def_goal_side_of_ball: +0.0219
- n_options_over_0_5: +0.0193
- att_width_m: +0.0177
- att_depth_m: +0.0092
- prev_type_recovery: +0.0080
- time_in_possession_s: +0.0075
- match_minute: +0.0060

## Caveats

- Only these matches are used, all from one anonymised provider sample. CIs reflect within-match sampling of possessions, not between-match or between-competition variation.
- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).
- Failed-pass labels depend on the documented Metrica event-mapping rules.
