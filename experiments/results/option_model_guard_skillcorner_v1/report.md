# Experiment `option_model_guard_skillcorner_v1`

- Code revision: `2c7c8f14ef64d0e47473eb93100519762698b0b4`
- Matches: skillcorner-1874553, skillcorner-1886347, skillcorner-1899585, skillcorner-1925299, skillcorner-1927964, skillcorner-1953632, skillcorner-1959846, skillcorner-1986691, skillcorner-1996435, skillcorner-1996436, skillcorner-2006229, skillcorner-2006363, skillcorner-2007448, skillcorner-2007721, skillcorner-2010085, skillcorner-2011166, skillcorner-2013725, skillcorner-2015213, skillcorner-2016236, skillcorner-2017461 (leave-one-match-out)
- Attempted passes: 15893 (completion rate 0.841); label sources {'BALL LOST': 2125, 'BALL OUT': 407, 'PASS': 13361}
- Receiver inference (completed passes): 0.748 top-1 (n=13361)
- Runtime: 252.5 s

## Pooled out-of-fold metrics

| model | family | log loss | Brier | AUC | ECE |
|---|---|---|---|---|---|
| option_v2_logreg | option | 0.3537 | 0.1113 | 0.810 | 0.021 |
| base_rate | baseline | 0.4390 | 0.1340 | 0.433 | 0.000 |

## Per-fold log loss (test match)

| model | skillcorner-1874553 | skillcorner-1886347 | skillcorner-1899585 | skillcorner-1925299 | skillcorner-1927964 | skillcorner-1953632 | skillcorner-1959846 | skillcorner-1986691 | skillcorner-1996435 | skillcorner-1996436 | skillcorner-2006229 | skillcorner-2006363 | skillcorner-2007448 | skillcorner-2007721 | skillcorner-2010085 | skillcorner-2011166 | skillcorner-2013725 | skillcorner-2015213 | skillcorner-2016236 | skillcorner-2017461 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_rate | 0.4404 | 0.3960 | 0.4661 | 0.4047 | 0.3287 | 0.3708 | 0.4429 | 0.3882 | 0.4275 | 0.5005 | 0.4355 | 0.4132 | 0.4306 | 0.4491 | 0.4215 | 0.5127 | 0.5132 | 0.4715 | 0.5362 | 0.5183 |
| option_v2_logreg | 0.3463 | 0.3069 | 0.3977 | 0.3272 | 0.2307 | 0.2761 | 0.3622 | 0.3144 | 0.3598 | 0.4113 | 0.3440 | 0.3458 | 0.3454 | 0.3697 | 0.3189 | 0.4755 | 0.4158 | 0.3873 | 0.4300 | 0.4062 |

## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)

| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|
| base_rate | option_v2_logreg | -0.0852 [-0.0913, -0.0797] | -0.0227 [-0.0249, -0.0206] | +0.3773 [+0.3609, +0.3936] |

## Caveats

- Only these matches are used, all from one anonymised provider sample. CIs reflect within-match sampling of possessions, not between-match or between-competition variation.
- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).
- Failed-pass labels depend on the documented Metrica event-mapping rules.
