# Experiment `temporal_guard_skillcorner_v1_02s`

- Code revision: `c90e67e27e76d83a9626ed148ee7ba3acda20bab`
- Matches: skillcorner-1874553, skillcorner-1886347, skillcorner-1899585, skillcorner-1925299, skillcorner-1927964, skillcorner-1953632, skillcorner-1959846, skillcorner-1986691, skillcorner-1996435, skillcorner-1996436, skillcorner-2006229, skillcorner-2006363, skillcorner-2007448, skillcorner-2007721, skillcorner-2010085, skillcorner-2011166, skillcorner-2013725, skillcorner-2015213, skillcorner-2016236, skillcorner-2017461 (leave-one-match-out)
- Attempted passes: 15893 (completion rate 0.841); label sources {'BALL LOST': 2125, 'BALL OUT': 407, 'PASS': 13361}
- Receiver inference (completed passes): 0.748 top-1 (n=13361)
- Runtime: 2866.0 s

## Pooled out-of-fold metrics

| model | family | log loss | Brier | AUC | ECE |
|---|---|---|---|---|---|
| Bdt_tt_gbm | temporal | 0.3746 | 0.1166 | 0.771 | 0.015 |
| Bdt_tob_gbm | temporal | 0.3835 | 0.1190 | 0.753 | 0.014 |
| Bdt_tt_logreg | temporal | 0.3874 | 0.1198 | 0.747 | 0.017 |
| Bdt_tob_logreg | temporal | 0.3965 | 0.1226 | 0.726 | 0.014 |
| Bdt_logreg | B-decision | 0.3969 | 0.1228 | 0.725 | 0.015 |
| Adt_tob_logreg | temporal | 0.4161 | 0.1277 | 0.665 | 0.005 |
| Adt_logreg | A-decision | 0.4183 | 0.1285 | 0.661 | 0.006 |
| base_rate | baseline | 0.4390 | 0.1340 | 0.433 | 0.000 |

## Per-fold log loss (test match)

| model | skillcorner-1874553 | skillcorner-1886347 | skillcorner-1899585 | skillcorner-1925299 | skillcorner-1927964 | skillcorner-1953632 | skillcorner-1959846 | skillcorner-1986691 | skillcorner-1996435 | skillcorner-1996436 | skillcorner-2006229 | skillcorner-2006363 | skillcorner-2007448 | skillcorner-2007721 | skillcorner-2010085 | skillcorner-2011166 | skillcorner-2013725 | skillcorner-2015213 | skillcorner-2016236 | skillcorner-2017461 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_rate | 0.4404 | 0.3960 | 0.4661 | 0.4047 | 0.3287 | 0.3708 | 0.4429 | 0.3882 | 0.4275 | 0.5005 | 0.4355 | 0.4132 | 0.4306 | 0.4491 | 0.4215 | 0.5127 | 0.5132 | 0.4715 | 0.5362 | 0.5183 |
| Adt_logreg | 0.4168 | 0.3640 | 0.4550 | 0.3944 | 0.3002 | 0.3418 | 0.4291 | 0.3636 | 0.4069 | 0.4762 | 0.4239 | 0.4022 | 0.4068 | 0.4290 | 0.4044 | 0.4914 | 0.4865 | 0.4478 | 0.5184 | 0.4950 |
| Bdt_logreg | 0.3942 | 0.3323 | 0.4378 | 0.3662 | 0.2757 | 0.3185 | 0.4087 | 0.3513 | 0.3882 | 0.4607 | 0.3984 | 0.3858 | 0.3952 | 0.4060 | 0.3734 | 0.4883 | 0.4706 | 0.4238 | 0.4865 | 0.4738 |
| Adt_tob_logreg | 0.4207 | 0.3646 | 0.4558 | 0.3905 | 0.3003 | 0.3408 | 0.4237 | 0.3654 | 0.4029 | 0.4703 | 0.4229 | 0.4014 | 0.4032 | 0.4275 | 0.3998 | 0.4899 | 0.4816 | 0.4390 | 0.5143 | 0.4926 |
| Bdt_tob_logreg | 0.3980 | 0.3335 | 0.4377 | 0.3643 | 0.2756 | 0.3185 | 0.4064 | 0.3538 | 0.3858 | 0.4609 | 0.3982 | 0.3878 | 0.3943 | 0.4058 | 0.3715 | 0.4872 | 0.4685 | 0.4198 | 0.4855 | 0.4729 |
| Bdt_tt_logreg | 0.3901 | 0.3264 | 0.4170 | 0.3641 | 0.2664 | 0.3079 | 0.3965 | 0.3462 | 0.3775 | 0.4502 | 0.3896 | 0.3789 | 0.3860 | 0.4016 | 0.3628 | 0.4757 | 0.4583 | 0.4088 | 0.4758 | 0.4600 |
| Bdt_tob_gbm | 0.3895 | 0.3290 | 0.4385 | 0.3573 | 0.2580 | 0.3059 | 0.3881 | 0.3319 | 0.3726 | 0.4368 | 0.3871 | 0.3709 | 0.3736 | 0.3934 | 0.3531 | 0.4715 | 0.4496 | 0.4082 | 0.4770 | 0.4758 |
| Bdt_tt_gbm | 0.3778 | 0.3231 | 0.4205 | 0.3533 | 0.2484 | 0.3008 | 0.3821 | 0.3244 | 0.3668 | 0.4310 | 0.3798 | 0.3619 | 0.3671 | 0.3857 | 0.3434 | 0.4599 | 0.4420 | 0.3935 | 0.4633 | 0.4607 |

## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)

| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|
| Adt_logreg | Bdt_logreg | -0.0214 [-0.0245, -0.0180] | -0.0057 [-0.0067, -0.0046] | +0.0640 [+0.0549, +0.0729] |
| Adt_logreg | Adt_tob_logreg | -0.0022 [-0.0034, -0.0012] | -0.0008 [-0.0012, -0.0005] | +0.0044 [-0.0006, +0.0103] |
| Bdt_tob_logreg | Bdt_tt_logreg | -0.0091 [-0.0113, -0.0071] | -0.0028 [-0.0035, -0.0020] | +0.0212 [+0.0165, +0.0260] |
| Bdt_tob_gbm | Bdt_tt_gbm | -0.0089 [-0.0110, -0.0069] | -0.0024 [-0.0031, -0.0018] | +0.0180 [+0.0133, +0.0227] |
| Adt_tob_logreg | Bdt_tob_logreg | -0.0196 [-0.0226, -0.0164] | -0.0051 [-0.0061, -0.0041] | +0.0600 [+0.0508, +0.0687] |

## Held-out permutation importance: Bdt_tt_logreg (Δ log loss, top 10)

- dist_to_goal: +0.5180
- ball_x_att: +0.3670
- passer_fwd_m_l10: +0.0186
- is_goal_kick: +0.0132
- att_width_m: +0.0121
- prev_type_pass: +0.0087
- def_centroid_fwd_m_l10: +0.0075
- ball_travel_m_l10: +0.0066
- att_centroid_fwd_m_l10: +0.0066
- angle_to_goal: +0.0058

## Caveats

- Only these matches are used, all from one anonymised provider sample. CIs reflect within-match sampling of possessions, not between-match or between-competition variation.
- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).
- Failed-pass labels depend on the documented Metrica event-mapping rules.
