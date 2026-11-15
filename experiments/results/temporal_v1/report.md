# Experiment `temporal_v1`

- Code revision: `379b98eea585e3a4754166ac9bbc27bdc9e35a95`
- Matches: metrica-sample-1, metrica-sample-2, metrica-sample-3 (leave-one-match-out)
- Attempted passes: 3593 (completion rate 0.796); label sources {'BALL LOST': 598, 'BALL OUT': 136, 'PASS': 2859}
- Receiver inference (completed passes): 0.762 top-1 (n=2859)
- Runtime: 191.6 s

## Pooled out-of-fold metrics

| model | family | log loss | Brier | AUC | ECE |
|---|---|---|---|---|---|
| Bdt_tt_gbm | temporal | 0.4361 | 0.1392 | 0.760 | 0.027 |
| Bdt_tt_hc_logreg | temporal | 0.4364 | 0.1376 | 0.758 | 0.017 |
| Bdt_tt05_logreg | temporal | 0.4384 | 0.1390 | 0.754 | 0.017 |
| Bdt_ttonly_logreg | temporal | 0.4392 | 0.1391 | 0.753 | 0.019 |
| Bdt_tt_logreg | temporal | 0.4395 | 0.1392 | 0.753 | 0.017 |
| Bdt_tt20_logreg | temporal | 0.4456 | 0.1408 | 0.741 | 0.019 |
| Bdt_tob_gbm | temporal | 0.4491 | 0.1439 | 0.738 | 0.019 |
| Bdt_tob_hc_logreg | temporal | 0.4494 | 0.1424 | 0.733 | 0.014 |
| Bdt_gbm | B-decision | 0.4539 | 0.1456 | 0.728 | 0.020 |
| Bdt_tob_logreg | temporal | 0.4554 | 0.1453 | 0.722 | 0.017 |
| Bdt_logreg | B-decision | 0.4562 | 0.1457 | 0.721 | 0.020 |
| Adt_tob_gbm | temporal | 0.4698 | 0.1504 | 0.698 | 0.023 |
| Adt_gbm | A-decision | 0.4762 | 0.1522 | 0.683 | 0.028 |
| Adt_tob_logreg | temporal | 0.4778 | 0.1521 | 0.664 | 0.006 |
| Adt_logreg | A-decision | 0.4819 | 0.1533 | 0.648 | 0.014 |
| Adt_te_logreg | temporal | 0.4830 | 0.1534 | 0.647 | 0.014 |
| base_rate | baseline | 0.5080 | 0.1631 | 0.463 | 0.028 |

## Per-fold log loss (test match)

| model | metrica-sample-1 | metrica-sample-2 | metrica-sample-3 |
|---|---|---|---|
| base_rate | 0.5550 | 0.4750 | 0.5009 |
| Adt_logreg | 0.5364 | 0.4389 | 0.4777 |
| Adt_gbm | 0.5133 | 0.4452 | 0.4747 |
| Bdt_logreg | 0.5128 | 0.4145 | 0.4492 |
| Bdt_gbm | 0.5016 | 0.4182 | 0.4486 |
| Adt_te_logreg | 0.5395 | 0.4388 | 0.4783 |
| Adt_tob_logreg | 0.5336 | 0.4336 | 0.4737 |
| Bdt_tob_logreg | 0.5126 | 0.4128 | 0.4488 |
| Bdt_tt_logreg | 0.4828 | 0.3976 | 0.4426 |
| Bdt_tt05_logreg | 0.4825 | 0.3996 | 0.4383 |
| Bdt_tt20_logreg | 0.4936 | 0.4036 | 0.4454 |
| Bdt_ttonly_logreg | 0.4843 | 0.3971 | 0.4412 |
| Bdt_tob_hc_logreg | 0.5021 | 0.4106 | 0.4431 |
| Bdt_tt_hc_logreg | 0.4774 | 0.3969 | 0.4391 |
| Adt_tob_gbm | 0.5137 | 0.4334 | 0.4678 |
| Bdt_tob_gbm | 0.4979 | 0.4128 | 0.4434 |
| Bdt_tt_gbm | 0.4791 | 0.3986 | 0.4358 |

## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)

| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|
| Adt_logreg | Adt_te_logreg | +0.0011 [-0.0000, +0.0026] | +0.0001 [-0.0001, +0.0003] | -0.0008 [-0.0033, +0.0016] |
| Adt_logreg | Adt_tob_logreg | -0.0041 [-0.0078, -0.0006] | -0.0013 [-0.0025, -0.0001] | +0.0160 [+0.0052, +0.0269] |
| Bdt_logreg | Bdt_tob_logreg | -0.0008 [-0.0030, +0.0013] | -0.0004 [-0.0012, +0.0004] | +0.0012 [-0.0025, +0.0050] |
| Bdt_tob_logreg | Bdt_tt_logreg | -0.0159 [-0.0234, -0.0089] | -0.0061 [-0.0089, -0.0035] | +0.0313 [+0.0208, +0.0426] |
| Bdt_logreg | Bdt_ttonly_logreg | -0.0170 [-0.0246, -0.0099] | -0.0065 [-0.0093, -0.0039] | +0.0326 [+0.0220, +0.0438] |
| Adt_tob_logreg | Bdt_tob_logreg | -0.0224 [-0.0304, -0.0145] | -0.0068 [-0.0097, -0.0040] | +0.0584 [+0.0419, +0.0756] |
| Adt_tob_logreg | Bdt_tt_logreg | -0.0383 [-0.0492, -0.0272] | -0.0129 [-0.0168, -0.0090] | +0.0897 [+0.0710, +0.1095] |
| Bdt_tt_logreg | Bdt_tt05_logreg | -0.0011 [-0.0042, +0.0019] | -0.0001 [-0.0013, +0.0010] | +0.0012 [-0.0031, +0.0055] |
| Bdt_tt_logreg | Bdt_tt20_logreg | +0.0061 [+0.0024, +0.0101] | +0.0016 [+0.0002, +0.0031] | -0.0127 [-0.0184, -0.0070] |
| Adt_tob_gbm | Bdt_tob_gbm | -0.0207 [-0.0285, -0.0124] | -0.0065 [-0.0092, -0.0036] | +0.0397 [+0.0249, +0.0533] |
| Bdt_tob_gbm | Bdt_tt_gbm | -0.0130 [-0.0199, -0.0066] | -0.0047 [-0.0073, -0.0021] | +0.0218 [+0.0120, +0.0323] |
| Adt_logreg | Bdt_logreg | -0.0257 [-0.0342, -0.0177] | -0.0077 [-0.0106, -0.0048] | +0.0732 [+0.0558, +0.0919] |
| Bdt_tob_hc_logreg | Bdt_tt_hc_logreg | -0.0131 [-0.0202, -0.0067] | -0.0048 [-0.0074, -0.0024] | +0.0258 [+0.0160, +0.0363] |

## Held-out permutation importance: Bdt_tt_logreg (Δ log loss, top 10)

- dist_to_goal: +0.1343
- ball_x_att: +0.0280
- passer_fwd_m_l10: +0.0258
- ball_travel_m_l10: +0.0250
- att_width_m: +0.0204
- def_line_depth_change_m_l10: +0.0165
- prev_type_pass: +0.0156
- att_depth_m: +0.0154
- n_options_over_0_5: +0.0126
- press_def_within_5m: +0.0085

## Caveats

- Only these matches are used, all from one anonymised provider sample. CIs reflect within-match sampling of possessions, not between-match or between-competition variation.
- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).
- Failed-pass labels depend on the documented Metrica event-mapping rules.
