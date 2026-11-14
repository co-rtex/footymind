# Experiment `option_model_guard_v1`

- Code revision: `2c7c8f14ef64d0e47473eb93100519762698b0b4`
- Matches: metrica-sample-1, metrica-sample-2, metrica-sample-3 (leave-one-match-out)
- Attempted passes: 3592 (completion rate 0.796); label sources {'BALL LOST': 598, 'BALL OUT': 136, 'PASS': 2858}
- Receiver inference (completed passes): 0.757 top-1 (n=2858)
- Runtime: 59.4 s

## Pooled out-of-fold metrics

| model | family | log loss | Brier | AUC | ECE |
|---|---|---|---|---|---|
| option_v2_logreg | option | 0.4187 | 0.1335 | 0.790 | 0.028 |
| base_rate | baseline | 0.5081 | 0.1631 | 0.462 | 0.028 |

## Per-fold log loss (test match)

| model | metrica-sample-1 | metrica-sample-2 | metrica-sample-3 |
|---|---|---|---|
| base_rate | 0.5554 | 0.4750 | 0.5009 |
| option_v2_logreg | 0.4651 | 0.3893 | 0.4091 |

## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)

| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|
| base_rate | option_v2_logreg | -0.0894 [-0.1031, -0.0750] | -0.0296 [-0.0349, -0.0244] | +0.3279 [+0.2973, +0.3598] |

## Caveats

- Only these matches are used, all from one anonymised provider sample. CIs reflect within-match sampling of possessions, not between-match or between-competition variation.
- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).
- Failed-pass labels depend on the documented Metrica event-mapping rules.
