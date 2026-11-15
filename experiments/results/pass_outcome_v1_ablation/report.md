# Experiment `pass_outcome_v1_ablation`

- Code revision: `a9f8c81b81cfe573f50a27de5361060d92956b33`
- Matches: metrica-sample-1, metrica-sample-2 (leave-one-match-out)
- Attempted passes: 2236 (completion rate 0.788); label sources {'BALL LOST': 387, 'BALL OUT': 86, 'PASS': 1763}
- Receiver inference (completed passes): 0.786 top-1 (n=1763)
- Runtime: 82.0 s

## Pooled out-of-fold metrics

| model | family | log loss | Brier | AUC | ECE |
|---|---|---|---|---|---|
| B_logreg | B | 0.3691 | 0.1165 | 0.863 | 0.029 |
| abl_A+target_geometry+target_defense_logreg | ablation | 0.3716 | 0.1177 | 0.854 | 0.032 |
| abl_A+target_geometry_logreg | ablation | 0.3793 | 0.1177 | 0.840 | 0.016 |
| abl_B_minus_target_geometry_logreg | ablation | 0.4214 | 0.1374 | 0.810 | 0.037 |
| abl_A+defender_only_logreg | ablation | 0.4227 | 0.1380 | 0.805 | 0.045 |
| abl_A+ray_defense_logreg | ablation | 0.4296 | 0.1402 | 0.800 | 0.043 |
| abl_A+pressure_logreg | ablation | 0.4311 | 0.1398 | 0.786 | 0.040 |
| A_logreg | A | 0.4377 | 0.1420 | 0.777 | 0.036 |

## Per-fold log loss (test match)

| model | metrica-sample-1 | metrica-sample-2 |
|---|---|---|
| A_logreg | 0.4681 | 0.4105 |
| B_logreg | 0.4115 | 0.3311 |
| abl_A+pressure_logreg | 0.4634 | 0.4023 |
| abl_A+ray_defense_logreg | 0.4647 | 0.3982 |
| abl_A+defender_only_logreg | 0.4598 | 0.3895 |
| abl_A+target_geometry_logreg | 0.4091 | 0.3526 |
| abl_A+target_geometry+target_defense_logreg | 0.4053 | 0.3415 |
| abl_B_minus_target_geometry_logreg | 0.4531 | 0.3929 |

## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)

| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|
| A_logreg | abl_A+pressure_logreg | -0.0065 [-0.0125, -0.0008] | -0.0022 [-0.0044, -0.0001] | +0.0084 [+0.0005, +0.0167] |
| A_logreg | abl_A+ray_defense_logreg | -0.0080 [-0.0166, +0.0003] | -0.0017 [-0.0050, +0.0014] | +0.0224 [+0.0118, +0.0325] |
| A_logreg | abl_A+defender_only_logreg | -0.0150 [-0.0250, -0.0047] | -0.0040 [-0.0079, -0.0002] | +0.0279 [+0.0142, +0.0419] |
| A_logreg | abl_A+target_geometry_logreg | -0.0584 [-0.0739, -0.0435] | -0.0242 [-0.0299, -0.0186] | +0.0623 [+0.0466, +0.0784] |
| abl_A+target_geometry_logreg | abl_A+target_geometry+target_defense_logreg | -0.0077 [-0.0190, +0.0036] | -0.0001 [-0.0039, +0.0039] | +0.0146 [+0.0032, +0.0255] |
| abl_A+target_geometry_logreg | B_logreg | -0.0102 [-0.0244, +0.0032] | -0.0012 [-0.0060, +0.0033] | +0.0230 [+0.0104, +0.0361] |
| A_logreg | abl_B_minus_target_geometry_logreg | -0.0163 [-0.0297, -0.0026] | -0.0046 [-0.0096, +0.0006] | +0.0324 [+0.0173, +0.0483] |

## Caveats

- Only these matches are used, all from one anonymised provider sample. CIs reflect within-match sampling of possessions, not between-match or between-competition variation.
- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).
- Failed-pass labels depend on the documented Metrica event-mapping rules.
