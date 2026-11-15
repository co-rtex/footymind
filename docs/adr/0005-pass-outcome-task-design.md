# ADR-0005: Pass-outcome task design and leakage controls

- Status: accepted (2026-11-13)

## Decision
The first research task is to predict the outcome of *attempted* passes (observed actions
only; hypothetical passes are never labelled). Features are limited to the decision frame,
causal velocities, the release direction/speed (≤ 0.12 s, truncated before the outcome)
and prior events. End locations are excluded because, for failed passes, Metrica records the
loss location, which would leak the label. Evaluation is leave-one-match-out with a-priori
hyperparameters and a possession-clustered paired bootstrap.

## Consequences
The intended target of failed passes is unobserved, so spatial models infer a candidate
receiver from the release direction (78.6% top-1 on completed passes). Conclusions are
limited to two matches from one anonymised provider sample.
