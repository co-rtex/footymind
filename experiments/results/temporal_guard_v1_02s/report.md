# Experiment `temporal_guard_v1_02s`

- Code revision: `379b98eea585e3a4754166ac9bbc27bdc9e35a95`
- Matches: metrica-sample-1, metrica-sample-2, metrica-sample-3 (leave-one-match-out)
- Attempted passes: 3592 (completion rate 0.796); label sources {'BALL LOST': 598, 'BALL OUT': 136, 'PASS': 2858}
- Receiver inference (completed passes): 0.757 top-1 (n=2858)
- Runtime: 803.5 s

## Pooled out-of-fold metrics

| model | family | log loss | Brier | AUC | ECE |
|---|---|---|---|---|---|
| Bdt_tt_gbm | temporal | 0.4372 | 0.1393 | 0.756 | 0.021 |
| Bdt_tt_logreg | temporal | 0.4420 | 0.1403 | 0.750 | 0.019 |
| Bdt_tob_gbm | temporal | 0.4498 | 0.1445 | 0.738 | 0.018 |
| Bdt_tob_logreg | temporal | 0.4557 | 0.1456 | 0.721 | 0.017 |
| Bdt_logreg | B-decision | 0.4563 | 0.1459 | 0.721 | 0.021 |
| Adt_tob_logreg | temporal | 0.4798 | 0.1527 | 0.654 | 0.009 |
| Adt_logreg | A-decision | 0.4821 | 0.1534 | 0.646 | 0.013 |
| base_rate | baseline | 0.5081 | 0.1631 | 0.462 | 0.028 |

## Per-fold log loss (test match)

| model | metrica-sample-1 | metrica-sample-2 | metrica-sample-3 |
|---|---|---|---|
| base_rate | 0.5554 | 0.4750 | 0.5009 |
| Adt_logreg | 0.5361 | 0.4391 | 0.4783 |
| Bdt_logreg | 0.5090 | 0.4170 | 0.4503 |
| Adt_tob_logreg | 0.5331 | 0.4362 | 0.4771 |
| Bdt_tob_logreg | 0.5070 | 0.4170 | 0.4503 |
| Bdt_tt_logreg | 0.4814 | 0.4016 | 0.4467 |
| Bdt_tob_gbm | 0.4945 | 0.4190 | 0.4426 |
| Bdt_tt_gbm | 0.4802 | 0.4036 | 0.4335 |

## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)

| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|
| Adt_logreg | Bdt_logreg | -0.0258 [-0.0344, -0.0176] | -0.0075 [-0.0106, -0.0047] | +0.0749 [+0.0573, +0.0933] |
| Adt_logreg | Adt_tob_logreg | -0.0023 [-0.0051, +0.0005] | -0.0007 [-0.0017, +0.0002] | +0.0078 [-0.0016, +0.0172] |
| Bdt_tob_logreg | Bdt_tt_logreg | -0.0137 [-0.0206, -0.0067] | -0.0053 [-0.0079, -0.0027] | +0.0287 [+0.0189, +0.0392] |
| Bdt_tob_gbm | Bdt_tt_gbm | -0.0126 [-0.0194, -0.0060] | -0.0051 [-0.0076, -0.0027] | +0.0183 [+0.0089, +0.0284] |
| Adt_tob_logreg | Bdt_tob_logreg | -0.0241 [-0.0322, -0.0161] | -0.0071 [-0.0100, -0.0044] | +0.0674 [+0.0503, +0.0854] |

## Held-out permutation importance: Bdt_tt_logreg (Δ log loss, top 10)

- dist_to_goal: +0.1518
- ball_x_att: +0.0275
- passer_fwd_m_l10: +0.0257
- ball_travel_m_l10: +0.0233
- att_width_m: +0.0181
- prev_type_pass: +0.0159
- press_nearest_def_change_m_l10: +0.0126
- def_line_depth_change_m_l10: +0.0119
- att_depth_m: +0.0114
- n_options_over_0_5: +0.0102

## Caveats

- Only these matches are used, all from one anonymised provider sample. CIs reflect within-match sampling of possessions, not between-match or between-competition variation.
- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).
- Failed-pass labels depend on the documented Metrica event-mapping rules.
