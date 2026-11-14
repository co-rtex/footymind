# Experiment `temporal_guard_skillcorner_v2_ballcensor`

- Code revision: `66c16db6459f2e30094778d5fe3de52b3240890c`
- Matches: skillcorner-1874553, skillcorner-1886347, skillcorner-1899585, skillcorner-1925299, skillcorner-1927964, skillcorner-1953632, skillcorner-1959846, skillcorner-1986691, skillcorner-1996435, skillcorner-1996436, skillcorner-2006229, skillcorner-2006363, skillcorner-2007448, skillcorner-2007721, skillcorner-2010085, skillcorner-2011166, skillcorner-2013725, skillcorner-2015213, skillcorner-2016236, skillcorner-2017461 (leave-one-match-out)
- Attempted passes: 15893 (completion rate 0.841); label sources {'BALL LOST': 2125, 'BALL OUT': 407, 'PASS': 13361}
- Receiver inference (completed passes): 0.748 top-1 (n=13361)
- Runtime: 1110.8 s

## Pooled out-of-fold metrics

| model | family | log loss | Brier | AUC | ECE |
|---|---|---|---|---|---|
| Bdt_tt_gbm | temporal | 0.3751 | 0.1167 | 0.770 | 0.015 |
| Bdt_tob_gbm | temporal | 0.3838 | 0.1192 | 0.753 | 0.013 |
| Bdt_tt_logreg | temporal | 0.3874 | 0.1198 | 0.747 | 0.017 |
| Bdt_tob_logreg | temporal | 0.3965 | 0.1226 | 0.725 | 0.014 |
| Bdt_logreg | B-decision | 0.3969 | 0.1228 | 0.725 | 0.015 |
| Adt_tob_logreg | temporal | 0.4158 | 0.1276 | 0.665 | 0.007 |
| Adt_logreg | A-decision | 0.4183 | 0.1285 | 0.661 | 0.006 |
| base_rate | baseline | 0.4390 | 0.1340 | 0.433 | 0.000 |

## Per-fold log loss (test match)

| model | skillcorner-1874553 | skillcorner-1886347 | skillcorner-1899585 | skillcorner-1925299 | skillcorner-1927964 | skillcorner-1953632 | skillcorner-1959846 | skillcorner-1986691 | skillcorner-1996435 | skillcorner-1996436 | skillcorner-2006229 | skillcorner-2006363 | skillcorner-2007448 | skillcorner-2007721 | skillcorner-2010085 | skillcorner-2011166 | skillcorner-2013725 | skillcorner-2015213 | skillcorner-2016236 | skillcorner-2017461 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_rate | 0.4404 | 0.3960 | 0.4661 | 0.4047 | 0.3287 | 0.3708 | 0.4429 | 0.3882 | 0.4275 | 0.5005 | 0.4355 | 0.4132 | 0.4306 | 0.4491 | 0.4215 | 0.5127 | 0.5132 | 0.4715 | 0.5362 | 0.5183 |
| Adt_logreg | 0.4168 | 0.3640 | 0.4550 | 0.3944 | 0.3002 | 0.3418 | 0.4291 | 0.3636 | 0.4069 | 0.4762 | 0.4239 | 0.4022 | 0.4068 | 0.4290 | 0.4044 | 0.4914 | 0.4865 | 0.4478 | 0.5184 | 0.4950 |
| Bdt_logreg | 0.3942 | 0.3323 | 0.4378 | 0.3662 | 0.2757 | 0.3185 | 0.4087 | 0.3513 | 0.3882 | 0.4607 | 0.3984 | 0.3858 | 0.3952 | 0.4060 | 0.3734 | 0.4883 | 0.4706 | 0.4238 | 0.4865 | 0.4738 |
| Adt_tob_logreg | 0.4197 | 0.3635 | 0.4554 | 0.3902 | 0.2993 | 0.3401 | 0.4228 | 0.3650 | 0.4025 | 0.4720 | 0.4227 | 0.4002 | 0.4034 | 0.4281 | 0.4011 | 0.4895 | 0.4809 | 0.4397 | 0.5150 | 0.4907 |
| Bdt_tob_logreg | 0.3977 | 0.3332 | 0.4373 | 0.3645 | 0.2756 | 0.3184 | 0.4061 | 0.3532 | 0.3859 | 0.4619 | 0.3982 | 0.3870 | 0.3948 | 0.4063 | 0.3723 | 0.4867 | 0.4683 | 0.4205 | 0.4861 | 0.4723 |
| Bdt_tt_logreg | 0.3902 | 0.3264 | 0.4167 | 0.3643 | 0.2665 | 0.3079 | 0.3965 | 0.3452 | 0.3778 | 0.4500 | 0.3898 | 0.3779 | 0.3862 | 0.4022 | 0.3633 | 0.4753 | 0.4577 | 0.4093 | 0.4765 | 0.4600 |
| Bdt_tob_gbm | 0.3888 | 0.3268 | 0.4375 | 0.3554 | 0.2594 | 0.3047 | 0.3897 | 0.3325 | 0.3738 | 0.4411 | 0.3862 | 0.3703 | 0.3751 | 0.3941 | 0.3569 | 0.4694 | 0.4508 | 0.4080 | 0.4800 | 0.4720 |
| Bdt_tt_gbm | 0.3829 | 0.3215 | 0.4197 | 0.3534 | 0.2477 | 0.3004 | 0.3817 | 0.3266 | 0.3681 | 0.4317 | 0.3806 | 0.3602 | 0.3673 | 0.3863 | 0.3443 | 0.4633 | 0.4427 | 0.3921 | 0.4646 | 0.4608 |

## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)

| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|
| Adt_logreg | Bdt_logreg | -0.0214 [-0.0245, -0.0180] | -0.0057 [-0.0067, -0.0046] | +0.0640 [+0.0549, +0.0729] |
| Adt_logreg | Adt_tob_logreg | -0.0025 [-0.0037, -0.0014] | -0.0009 [-0.0013, -0.0006] | +0.0044 [-0.0007, +0.0104] |
| Bdt_tob_logreg | Bdt_tt_logreg | -0.0091 [-0.0113, -0.0072] | -0.0028 [-0.0035, -0.0020] | +0.0212 [+0.0165, +0.0259] |
| Bdt_tob_gbm | Bdt_tt_gbm | -0.0087 [-0.0108, -0.0067] | -0.0025 [-0.0031, -0.0018] | +0.0171 [+0.0125, +0.0219] |
| Adt_tob_logreg | Bdt_tob_logreg | -0.0193 [-0.0223, -0.0161] | -0.0050 [-0.0060, -0.0040] | +0.0600 [+0.0511, +0.0687] |

## Held-out permutation importance: Bdt_tt_logreg (Δ log loss, top 10)

- dist_to_goal: +0.5173
- ball_x_att: +0.3664
- passer_fwd_m_l10: +0.0185
- is_goal_kick: +0.0131
- att_width_m: +0.0121
- prev_type_pass: +0.0087
- def_centroid_fwd_m_l10: +0.0076
- att_centroid_fwd_m_l10: +0.0066
- ball_travel_m_l10: +0.0065
- angle_to_goal: +0.0059

## Caveats

- Only these matches are used, all from one anonymised provider sample. CIs reflect within-match sampling of possessions, not between-match or between-competition variation.
- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).
- Failed-pass labels depend on the documented Metrica event-mapping rules.
