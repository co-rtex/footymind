# Experiment `pass_outcome_v2_sens_no_after_carry`

- Code revision: `dffc3e8a5ecaa4be7208b7f11defcfbcc651eed4`
- Matches: metrica-sample-1, metrica-sample-2, metrica-sample-3 (leave-one-match-out)
- Attempted passes: 2876 (completion rate 0.772); label sources {'BALL LOST': 527, 'BALL OUT': 128, 'PASS': 2221}
- Receiver inference (completed passes): 0.748 top-1 (n=2221)
- Runtime: 53.5 s

## Pooled out-of-fold metrics

| model | family | log loss | Brier | AUC | ECE |
|---|---|---|---|---|---|
| B_gbm | B | 0.3784 | 0.1229 | 0.858 | 0.027 |
| B_logreg | B | 0.3859 | 0.1232 | 0.856 | 0.030 |
| A_gbm | A | 0.4247 | 0.1392 | 0.808 | 0.018 |
| abl_A+defender_only_logreg | ablation | 0.4464 | 0.1460 | 0.790 | 0.032 |
| A_logreg | A | 0.4576 | 0.1494 | 0.772 | 0.040 |
| base_rate | baseline | 0.5425 | 0.1780 | 0.438 | 0.006 |

## Per-fold log loss (test match)

| model | metrica-sample-1 | metrica-sample-2 | metrica-sample-3 |
|---|---|---|---|
| base_rate | 0.5567 | 0.4921 | 0.6118 |
| A_logreg | 0.4657 | 0.4124 | 0.5277 |
| A_gbm | 0.4372 | 0.3742 | 0.4974 |
| B_logreg | 0.3963 | 0.3261 | 0.4788 |
| B_gbm | 0.3868 | 0.3255 | 0.4619 |
| abl_A+defender_only_logreg | 0.4541 | 0.3906 | 0.5364 |

## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)

| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|
| A_logreg | B_logreg | -0.0718 [-0.0879, -0.0564] | -0.0263 [-0.0321, -0.0209] | +0.0839 [+0.0687, +0.0996] |
| A_gbm | B_gbm | -0.0464 [-0.0607, -0.0332] | -0.0163 [-0.0214, -0.0112] | +0.0496 [+0.0360, +0.0636] |
| A_logreg | abl_A+defender_only_logreg | -0.0113 [-0.0185, -0.0040] | -0.0034 [-0.0063, -0.0007] | +0.0178 [+0.0085, +0.0272] |

## Caveats

- Only these matches are used, all from one anonymised provider sample. CIs reflect within-match sampling of possessions, not between-match or between-competition variation.
- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).
- Failed-pass labels depend on the documented Metrica event-mapping rules.
