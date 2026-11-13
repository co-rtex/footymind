# Experiment `option_model_v2_lead`

- Code revision: `8fefc1102ac67b51794755842940f89fbcc5373f`
- Matches: metrica-sample-1, metrica-sample-2, metrica-sample-3 (leave-one-match-out)
- Attempted passes: 3633 (completion rate 0.792); label sources {'BALL LOST': 617, 'BALL OUT': 137, 'PASS': 2879}
- Receiver inference (completed passes): 0.761 top-1 (n=2879)
- Runtime: 54.2 s

## Pooled out-of-fold metrics

| model | family | log loss | Brier | AUC | ECE |
|---|---|---|---|---|---|
| B_logreg | B | 0.3580 | 0.1133 | 0.866 | 0.026 |
| option_v2_logreg | option | 0.4176 | 0.1335 | 0.796 | 0.020 |
| option_logreg | option | 0.4221 | 0.1351 | 0.789 | 0.017 |
| A_logreg | A | 0.4290 | 0.1379 | 0.784 | 0.030 |

## Per-fold log loss (test match)

| model | metrica-sample-1 | metrica-sample-2 | metrica-sample-3 |
|---|---|---|---|
| A_logreg | 0.4691 | 0.4021 | 0.4215 |
| option_logreg | 0.4674 | 0.3943 | 0.4113 |
| option_v2_logreg | 0.4608 | 0.3938 | 0.4052 |
| B_logreg | 0.3932 | 0.3195 | 0.3639 |

## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)

| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|
| option_logreg | option_v2_logreg | -0.0045 [-0.0081, -0.0010] | -0.0016 [-0.0030, -0.0001] | +0.0063 [+0.0019, +0.0106] |
| A_logreg | option_v2_logreg | -0.0114 [-0.0247, +0.0022] | -0.0044 [-0.0094, +0.0005] | +0.0118 [-0.0053, +0.0296] |

## Caveats

- Only these matches are used, all from one anonymised provider sample. CIs reflect within-match sampling of possessions, not between-match or between-competition variation.
- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).
- Failed-pass labels depend on the documented Metrica event-mapping rules.
