# Experiment `pass_outcome_skillcorner_v2_importance`

- Code revision: `bd074ecdde0410ecf6e0d0520c3b77c42e8d5cf2`
- Matches: skillcorner-1874553, skillcorner-1886347, skillcorner-1899585, skillcorner-1925299, skillcorner-1927964, skillcorner-1953632, skillcorner-1959846, skillcorner-1986691, skillcorner-1996435, skillcorner-1996436, skillcorner-2006229, skillcorner-2006363, skillcorner-2007448, skillcorner-2007721, skillcorner-2010085, skillcorner-2011166, skillcorner-2013725, skillcorner-2015213, skillcorner-2016236, skillcorner-2017461 (leave-one-match-out)
- Attempted passes: 15914 (completion rate 0.840); label sources {'BALL LOST': 2132, 'BALL OUT': 407, 'PASS': 13375}
- Receiver inference (completed passes): 0.752 top-1 (n=13375)
- Runtime: 178.5 s

## Pooled out-of-fold metrics

| model | family | log loss | Brier | AUC | ECE |
|---|---|---|---|---|---|
| Bdt_logreg | B-decision | 0.3968 | 0.1227 | 0.725 | 0.015 |
| Adt_logreg | A-decision | 0.4182 | 0.1286 | 0.663 | 0.009 |

## Per-fold log loss (test match)

| model | skillcorner-1874553 | skillcorner-1886347 | skillcorner-1899585 | skillcorner-1925299 | skillcorner-1927964 | skillcorner-1953632 | skillcorner-1959846 | skillcorner-1986691 | skillcorner-1996435 | skillcorner-1996436 | skillcorner-2006229 | skillcorner-2006363 | skillcorner-2007448 | skillcorner-2007721 | skillcorner-2010085 | skillcorner-2011166 | skillcorner-2013725 | skillcorner-2015213 | skillcorner-2016236 | skillcorner-2017461 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Adt_logreg | 0.4160 | 0.3635 | 0.4558 | 0.3959 | 0.3003 | 0.3421 | 0.4297 | 0.3636 | 0.4061 | 0.4741 | 0.4242 | 0.4011 | 0.4068 | 0.4285 | 0.4025 | 0.4903 | 0.4870 | 0.4486 | 0.5180 | 0.4963 |
| Bdt_logreg | 0.3943 | 0.3298 | 0.4375 | 0.3671 | 0.2762 | 0.3186 | 0.4099 | 0.3529 | 0.3879 | 0.4579 | 0.3966 | 0.3826 | 0.3948 | 0.4065 | 0.3684 | 0.4867 | 0.4728 | 0.4266 | 0.4883 | 0.4777 |

## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)

| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|
| Adt_logreg | Bdt_logreg | -0.0214 [-0.0248, -0.0181] | -0.0059 [-0.0070, -0.0048] | +0.0619 [+0.0528, +0.0716] |

## Held-out permutation importance: Bdt_logreg (Δ log loss, top 10)

- dist_to_goal: +0.4571
- ball_x_att: +0.3297
- def_goal_side_of_ball: +0.0200
- is_goal_kick: +0.0173
- prev_type_pass: +0.0118
- att_width_m: +0.0099
- prev_type_recovery: +0.0079
- angle_to_goal: +0.0074
- ball_y_abs: +0.0021
- press_closing_speed: +0.0015

## Caveats

- Only these matches are used, all from one anonymised provider sample. CIs reflect within-match sampling of possessions, not between-match or between-competition variation.
- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).
- Failed-pass labels depend on the documented Metrica event-mapping rules.
