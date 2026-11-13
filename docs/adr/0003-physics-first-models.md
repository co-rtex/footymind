# ADR-0003: Physics-based spatial models first; learned complexity must be earned

- Status: accepted (2026-11-13)

## Decision
The interactive engine uses a literature time-to-intercept pitch-control model (Spearman
2018) and transparent lane geometry, labelled as *uncalibrated estimates*. Learned models
are evaluated offline against event-only baselines under leave-one-match-out validation.
More expressive models (GNNs, generative trajectory models, RL) are not built until a
controlled experiment shows that simpler interaction models leave measurable signal
unexplained, and enough data exists to train them credibly.

## Evidence so far
`experiments/results/pass_outcome_v1`: on 2,236 attempted passes from 2 matches,
interaction features (C) did not significantly improve on spatial features (B)
(Δ log loss −0.0066 [−0.0157, +0.0026] for gradient boosting). This does not justify a GNN.
