# Experiment `pass_outcome_v1`

- Code revision: `a9f8c81b81cfe573f50a27de5361060d92956b33`
- Matches: metrica-sample-1, metrica-sample-2 (leave-one-match-out)
- Attempted passes: 2236 (completion rate 0.788); label sources {'BALL LOST': 387, 'BALL OUT': 86, 'PASS': 1763}
- Receiver inference (completed passes): 0.786 top-1 (n=1763)
- Runtime: 92.4 s

## Pooled out-of-fold metrics

| model | family | log loss | Brier | AUC | ECE |
|---|---|---|---|---|---|
| C_gbm | C | 0.3649 | 0.1177 | 0.861 | 0.036 |
| B_logreg | B | 0.3691 | 0.1165 | 0.863 | 0.029 |
| C_logreg | C | 0.3713 | 0.1169 | 0.861 | 0.028 |
| B_gbm | B | 0.3715 | 0.1202 | 0.858 | 0.037 |
| abl_A+target_geometry+target_defense_logreg | ablation | 0.3716 | 0.1177 | 0.854 | 0.032 |
| abl_A+target_geometry_logreg | ablation | 0.3793 | 0.1177 | 0.840 | 0.016 |
| abl_B_minus_target_geometry_logreg | ablation | 0.4214 | 0.1374 | 0.810 | 0.037 |
| abl_A+defender_only_logreg | ablation | 0.4227 | 0.1380 | 0.805 | 0.045 |
| A_gbm | A | 0.4234 | 0.1371 | 0.797 | 0.026 |
| abl_A+ray_defense_logreg | ablation | 0.4296 | 0.1402 | 0.800 | 0.043 |
| abl_A+pressure_logreg | ablation | 0.4311 | 0.1398 | 0.786 | 0.040 |
| A_logreg | A | 0.4377 | 0.1420 | 0.777 | 0.036 |
| physics_platt | physics | 0.4637 | 0.1499 | 0.732 | 0.041 |
| base_rate | baseline | 0.5242 | 0.1695 | 0.455 | 0.060 |
| physics_raw | physics | 0.7838 | 0.2100 | 0.744 | 0.184 |

## Per-fold log loss (test match)

| model | metrica-sample-1 | metrica-sample-2 |
|---|---|---|
| base_rate | 0.5662 | 0.4865 |
| A_logreg | 0.4681 | 0.4105 |
| A_gbm | 0.4531 | 0.3968 |
| physics_raw | 0.8228 | 0.7489 |
| physics_platt | 0.4967 | 0.4343 |
| B_logreg | 0.4115 | 0.3311 |
| B_gbm | 0.4118 | 0.3354 |
| C_logreg | 0.4145 | 0.3326 |
| C_gbm | 0.4093 | 0.3251 |
| abl_A+pressure_logreg | 0.4634 | 0.4023 |
| abl_A+ray_defense_logreg | 0.4647 | 0.3982 |
| abl_A+defender_only_logreg | 0.4598 | 0.3895 |
| abl_A+target_geometry_logreg | 0.4091 | 0.3526 |
| abl_A+target_geometry+target_defense_logreg | 0.4053 | 0.3415 |
| abl_B_minus_target_geometry_logreg | 0.4531 | 0.3929 |

## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)

| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|
| base_rate | A_logreg | -0.0865 [-0.1025, -0.0708] | -0.0276 [-0.0340, -0.0213] | +0.3226 [+0.2891, +0.3564] |
| A_logreg | B_logreg | -0.0686 [-0.0893, -0.0499] | -0.0255 [-0.0327, -0.0189] | +0.0853 [+0.0676, +0.1052] |
| A_gbm | B_gbm | -0.0519 [-0.0716, -0.0327] | -0.0169 [-0.0241, -0.0103] | +0.0610 [+0.0425, +0.0801] |
| B_logreg | C_logreg | +0.0022 [-0.0058, +0.0102] | +0.0004 [-0.0024, +0.0034] | -0.0014 [-0.0075, +0.0048] |
| B_gbm | C_gbm | -0.0066 [-0.0157, +0.0026] | -0.0025 [-0.0055, +0.0008] | +0.0037 [-0.0031, +0.0102] |
| A_logreg | physics_platt | +0.0261 [+0.0088, +0.0433] | +0.0080 [+0.0013, +0.0150] | -0.0455 [-0.0736, -0.0174] |
| physics_platt | B_logreg | -0.0947 [-0.1168, -0.0748] | -0.0334 [-0.0415, -0.0263] | +0.1307 [+0.1065, +0.1558] |

## Held-out permutation importance: B_logreg (Δ log loss, top 10)

- cand_angle_dev: +0.1481
- dist_to_goal: +0.0619
- ray_def_in_cone: +0.0464
- ball_x_att: +0.0365
- prev_type_pass: +0.0310
- cand_p_success_physics: +0.0289
- cand_distance_m: +0.0264
- cand_def_within_3m_lane: +0.0186
- cand_ppcf: +0.0151
- match_minute: +0.0138

## Held-out permutation importance: C_gbm (Δ log loss, top 10)

- cand_angle_dev: +0.1131
- time_in_possession_s: +0.0154
- ray_min_def_perp_m: +0.0138
- cand2_p_success_physics: +0.0107
- sum_p_intercept_ray: +0.0094
- release_cos: +0.0089
- cand_p_lane_clear: +0.0075
- cand_distance_m: +0.0044
- att_width_m: +0.0044
- release_speed: +0.0036

## Caveats

- Only these matches are used, all from one anonymised provider sample. CIs reflect within-match sampling of possessions, not between-match or between-competition variation.
- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).
- Failed-pass labels depend on the documented Metrica event-mapping rules.
