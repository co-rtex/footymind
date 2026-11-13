# Experiment `pass_outcome_v2_3match`

- Code revision: `dffc3e8a5ecaa4be7208b7f11defcfbcc651eed4`
- Matches: metrica-sample-1, metrica-sample-2, metrica-sample-3 (leave-one-match-out)
- Attempted passes: 3633 (completion rate 0.792); label sources {'BALL LOST': 617, 'BALL OUT': 137, 'PASS': 2879}
- Receiver inference (completed passes): 0.761 top-1 (n=2879)
- Runtime: 97.0 s

## Pooled out-of-fold metrics

| model | family | log loss | Brier | AUC | ECE |
|---|---|---|---|---|---|
| C_gbm | C | 0.3486 | 0.1112 | 0.869 | 0.017 |
| B_gbm | B | 0.3502 | 0.1121 | 0.868 | 0.019 |
| B_logreg | B | 0.3580 | 0.1133 | 0.866 | 0.026 |
| C_logreg | C | 0.3621 | 0.1143 | 0.862 | 0.024 |
| abl_A+target_geometry_logreg | ablation | 0.3819 | 0.1190 | 0.837 | 0.019 |
| A_gbm | A | 0.4002 | 0.1294 | 0.817 | 0.016 |
| abl_A+defender_only_logreg | ablation | 0.4146 | 0.1339 | 0.805 | 0.027 |
| A_logreg | A | 0.4290 | 0.1379 | 0.784 | 0.030 |
| physics_platt | physics | 0.4557 | 0.1474 | 0.748 | 0.048 |
| base_rate | baseline | 0.5126 | 0.1651 | 0.461 | 0.030 |
| physics_raw | physics | 0.7716 | 0.2059 | 0.751 | 0.174 |

## Per-fold log loss (test match)

| model | metrica-sample-1 | metrica-sample-2 | metrica-sample-3 |
|---|---|---|---|
| base_rate | 0.5627 | 0.4800 | 0.5023 |
| A_logreg | 0.4691 | 0.4021 | 0.4215 |
| A_gbm | 0.4375 | 0.3712 | 0.3965 |
| physics_raw | 0.8228 | 0.7489 | 0.7521 |
| physics_platt | 0.4942 | 0.4312 | 0.4472 |
| B_logreg | 0.3932 | 0.3195 | 0.3639 |
| B_gbm | 0.3771 | 0.3221 | 0.3537 |
| C_logreg | 0.3951 | 0.3190 | 0.3737 |
| C_gbm | 0.3755 | 0.3173 | 0.3548 |
| abl_A+defender_only_logreg | 0.4566 | 0.3792 | 0.4128 |
| abl_A+target_geometry_logreg | 0.4159 | 0.3469 | 0.3857 |

## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)

| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|
| A_logreg | B_logreg | -0.0710 [-0.0851, -0.0572] | -0.0247 [-0.0297, -0.0196] | +0.0821 [+0.0682, +0.0963] |
| A_gbm | B_gbm | -0.0500 [-0.0624, -0.0371] | -0.0173 [-0.0221, -0.0126] | +0.0513 [+0.0382, +0.0637] |
| B_gbm | C_gbm | -0.0016 [-0.0061, +0.0029] | -0.0009 [-0.0025, +0.0008] | +0.0009 [-0.0030, +0.0048] |
| A_logreg | abl_A+defender_only_logreg | -0.0144 [-0.0210, -0.0075] | -0.0040 [-0.0064, -0.0014] | +0.0216 [+0.0129, +0.0304] |
| A_logreg | abl_A+target_geometry_logreg | -0.0471 [-0.0587, -0.0354] | -0.0189 [-0.0232, -0.0146] | +0.0536 [+0.0411, +0.0658] |
| A_logreg | physics_platt | +0.0266 [+0.0133, +0.0410] | +0.0095 [+0.0044, +0.0151] | -0.0355 [-0.0575, -0.0135] |

## Held-out permutation importance: B_logreg (Δ log loss, top 10)

- cand_angle_dev: +0.1162
- cand_p_success_physics: +0.0968
- dist_to_goal: +0.0485
- cand_p_lane_clear: +0.0311
- ray_def_in_cone: +0.0275
- ball_x_att: +0.0233
- cand_def_within_3m_lane: +0.0169
- release_cos: +0.0156
- prev_type_pass: +0.0132
- cand_ppcf: +0.0117

## Caveats

- Only these matches are used, all from one anonymised provider sample. CIs reflect within-match sampling of possessions, not between-match or between-competition variation.
- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).
- Failed-pass labels depend on the documented Metrica event-mapping rules.
