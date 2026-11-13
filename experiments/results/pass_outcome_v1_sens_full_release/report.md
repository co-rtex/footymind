# Experiment `pass_outcome_v1_sens_full_release`

- Code revision: `a9f8c81b81cfe573f50a27de5361060d92956b33`
- Matches: metrica-sample-1, metrica-sample-2 (leave-one-match-out)
- Attempted passes: 2214 (completion rate 0.794); label sources {'BALL LOST': 370, 'BALL OUT': 86, 'PASS': 1758}
- Receiver inference (completed passes): 0.786 top-1 (n=1758)
- Runtime: 43.5 s

## Pooled out-of-fold metrics

| model | family | log loss | Brier | AUC | ECE |
|---|---|---|---|---|---|
| C_gbm | C | 0.3667 | 0.1183 | 0.857 | 0.037 |
| B_gbm | B | 0.3692 | 0.1196 | 0.857 | 0.038 |
| B_logreg | B | 0.3721 | 0.1176 | 0.858 | 0.035 |
| abl_A+defender_only_logreg | ablation | 0.4225 | 0.1377 | 0.801 | 0.050 |
| A_gbm | A | 0.4234 | 0.1377 | 0.791 | 0.034 |
| A_logreg | A | 0.4340 | 0.1405 | 0.776 | 0.037 |
| base_rate | baseline | 0.5164 | 0.1661 | 0.455 | 0.059 |

## Per-fold log loss (test match)

| model | metrica-sample-1 | metrica-sample-2 |
|---|---|---|
| base_rate | 0.5586 | 0.4790 |
| A_logreg | 0.4629 | 0.4084 |
| A_gbm | 0.4545 | 0.3958 |
| B_logreg | 0.4139 | 0.3349 |
| B_gbm | 0.4085 | 0.3343 |
| C_gbm | 0.4111 | 0.3272 |
| abl_A+defender_only_logreg | 0.4588 | 0.3902 |

## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)

| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|
| A_logreg | B_logreg | -0.0619 [-0.0827, -0.0425] | -0.0229 [-0.0301, -0.0163] | +0.0823 [+0.0633, +0.1023] |
| A_gbm | B_gbm | -0.0542 [-0.0731, -0.0346] | -0.0181 [-0.0250, -0.0114] | +0.0655 [+0.0470, +0.0844] |
| A_logreg | abl_A+defender_only_logreg | -0.0115 [-0.0221, -0.0009] | -0.0028 [-0.0068, +0.0009] | +0.0254 [+0.0114, +0.0391] |

## Caveats

- Only these matches are used, all from one anonymised provider sample. CIs reflect within-match sampling of possessions, not between-match or between-competition variation.
- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).
- Failed-pass labels depend on the documented Metrica event-mapping rules.
