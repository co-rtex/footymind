# Experiment `pass_outcome_v1_sens_no_ball_out`

- Code revision: `a9f8c81b81cfe573f50a27de5361060d92956b33`
- Matches: metrica-sample-1, metrica-sample-2 (leave-one-match-out)
- Attempted passes: 2150 (completion rate 0.820); label sources {'BALL LOST': 387, 'PASS': 1763}
- Receiver inference (completed passes): 0.786 top-1 (n=1763)
- Runtime: 42.6 s

## Pooled out-of-fold metrics

| model | family | log loss | Brier | AUC | ECE |
|---|---|---|---|---|---|
| C_gbm | C | 0.3395 | 0.1090 | 0.862 | 0.035 |
| B_logreg | B | 0.3423 | 0.1080 | 0.864 | 0.029 |
| B_gbm | B | 0.3449 | 0.1116 | 0.858 | 0.037 |
| abl_A+defender_only_logreg | ablation | 0.3749 | 0.1205 | 0.825 | 0.040 |
| A_gbm | A | 0.3899 | 0.1235 | 0.797 | 0.025 |
| A_logreg | A | 0.3984 | 0.1265 | 0.784 | 0.028 |
| physics_platt | physics | 0.4142 | 0.1311 | 0.759 | 0.033 |
| base_rate | baseline | 0.4798 | 0.1501 | 0.451 | 0.058 |

## Per-fold log loss (test match)

| model | metrica-sample-1 | metrica-sample-2 |
|---|---|---|
| base_rate | 0.5263 | 0.4385 |
| A_logreg | 0.4298 | 0.3704 |
| A_gbm | 0.4209 | 0.3623 |
| physics_platt | 0.4505 | 0.3818 |
| B_logreg | 0.3839 | 0.3052 |
| B_gbm | 0.3846 | 0.3096 |
| C_gbm | 0.3747 | 0.3081 |
| abl_A+defender_only_logreg | 0.4137 | 0.3404 |

## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)

| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|
| A_logreg | B_logreg | -0.0561 [-0.0755, -0.0375] | -0.0185 [-0.0254, -0.0119] | +0.0798 [+0.0610, +0.0999] |
| A_gbm | B_gbm | -0.0450 [-0.0651, -0.0269] | -0.0118 [-0.0189, -0.0056] | +0.0607 [+0.0411, +0.0825] |
| A_logreg | abl_A+defender_only_logreg | -0.0235 [-0.0350, -0.0120] | -0.0060 [-0.0100, -0.0021] | +0.0405 [+0.0256, +0.0557] |

## Caveats

- Only these matches are used, all from one anonymised provider sample. CIs reflect within-match sampling of possessions, not between-match or between-competition variation.
- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).
- Failed-pass labels depend on the documented Metrica event-mapping rules.
