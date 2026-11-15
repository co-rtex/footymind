# Experiment `pass_outcome_skillcorner_v1_importance`

- Code revision: `1bce367078cec43abc40cb90bf26091e6c1dcb40`
- Matches: skillcorner-1874553, skillcorner-1886347, skillcorner-1899585, skillcorner-1925299, skillcorner-1927964, skillcorner-1953632, skillcorner-1959846, skillcorner-1986691, skillcorner-1996435, skillcorner-1996436, skillcorner-2006229, skillcorner-2006363, skillcorner-2007448, skillcorner-2007721, skillcorner-2010085, skillcorner-2011166, skillcorner-2013725, skillcorner-2015213, skillcorner-2016236, skillcorner-2017461 (leave-one-match-out)
- Attempted passes: 15867 (completion rate 0.841); label sources {'BALL LOST': 2521, 'PASS': 13346}
- Receiver inference (completed passes): 0.724 top-1 (n=13346)
- Runtime: 187.8 s

## Pooled out-of-fold metrics

| model | family | log loss | Brier | AUC | ECE |
|---|---|---|---|---|---|
| Bdt_logreg | B-decision | 0.3999 | 0.1233 | 0.714 | 0.015 |
| Adt_logreg | A-decision | 0.4234 | 0.1299 | 0.641 | 0.007 |

## Per-fold log loss (test match)

| model | skillcorner-1874553 | skillcorner-1886347 | skillcorner-1899585 | skillcorner-1925299 | skillcorner-1927964 | skillcorner-1953632 | skillcorner-1959846 | skillcorner-1986691 | skillcorner-1996435 | skillcorner-1996436 | skillcorner-2006229 | skillcorner-2006363 | skillcorner-2007448 | skillcorner-2007721 | skillcorner-2010085 | skillcorner-2011166 | skillcorner-2013725 | skillcorner-2015213 | skillcorner-2016236 | skillcorner-2017461 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Adt_logreg | 0.4194 | 0.3783 | 0.4700 | 0.4023 | 0.3043 | 0.3499 | 0.4382 | 0.3649 | 0.4149 | 0.4811 | 0.4259 | 0.4030 | 0.4143 | 0.4302 | 0.4097 | 0.4884 | 0.4931 | 0.4541 | 0.5172 | 0.4938 |
| Bdt_logreg | 0.4010 | 0.3372 | 0.4493 | 0.3705 | 0.2803 | 0.3246 | 0.4129 | 0.3564 | 0.3937 | 0.4635 | 0.3966 | 0.3846 | 0.3972 | 0.4057 | 0.3714 | 0.4841 | 0.4754 | 0.4273 | 0.4848 | 0.4780 |

## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)

| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|
| Adt_logreg | Bdt_logreg | -0.0235 [-0.0272, -0.0201] | -0.0066 [-0.0078, -0.0055] | +0.0724 [+0.0622, +0.0826] |

## Held-out permutation importance: Bdt_logreg (Δ log loss, top 10)

- dist_to_goal: +0.4140
- ball_x_att: +0.3038
- def_goal_side_of_ball: +0.0182
- att_width_m: +0.0136
- angle_to_goal: +0.0067
- press_closing_speed: +0.0024
- prev_type_pass: +0.0021
- ball_y_abs: +0.0018
- n_options_over_0_5: +0.0016
- def_line_depth_m: +0.0010

## Caveats

- Only these matches are used, all from one anonymised provider sample. CIs reflect within-match sampling of possessions, not between-match or between-competition variation.
- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).
- Failed-pass labels depend on the documented Metrica event-mapping rules.
