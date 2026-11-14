# Off-ball opportunity under reactive defenders (session 4)

Status: **pre-registered** (this section was written and committed before the study was run
on full data; only a 2-match pilot was run, to check code paths and runtime, and its numbers
are not used). Code: `src/footymind/ghost/offball_opportunity.py` (definitions),
`src/footymind/research/offball_study.py` (study, decision rules). Results:
`experiments/results/offball_opportunity_v1/` (Metrica) and
`experiments/results/offball_opportunity_v1_skillcorner/`. Providers are never pooled.

## Question
For an attacker's observed off-ball movement over the last 1.0 s before a pass decision, read
at the 0.2 s guard frame (ADR-0007), how much does the team's *modelled* passing opportunity
change once plausible defender responses to that movement are accounted for, and how much
does the answer depend on the response assumption?

Background: `run_value` holds defenders at their observed positions and fails as a ranking
(docs/RESEARCH.md); the movement decomposition (C31/C32) shows per-pass credit depends on the
convention. The rule policies `marker_follows` and `lane_cover` passed an observational check
when driven by the attackers' *innovation* (C37 v2), so they are the best-supported way we have
to put a response back.

## Design
At the guard frame `t`, `V(S)` is the team's opportunity under the serving option model:
`forward` (primary: best progressive option, teammates > 5 m ahead of the ball), `sum`
(summed completion probability) and `own` (the option to the attacker).

Counterfactual for attacker `a`: only `a` is placed at their observed position and causal
velocity 1 s earlier. Change = `V(factual) − V(counterfactual)` under four defender assumptions:

| assumption | defenders in the counterfactual | factual |
|---|---|---|
| `static` | observed at `t` (the `run_value` convention) | observed |
| `constant_velocity` | every outfield defender at `x(t−1) + v(t−1)·1 s`: nobody reacted to anything | the same background with `a` observed |
| `marker_follows` | observed minus the rule's response to `a`'s movement | observed |
| `lane_cover` | observed minus the rule's response to `a`'s movement (kinematically clipped) | observed |

**Attributed movement.** The policies respond to `a`'s **innovation**, `x(t) − x(t−1) − v(t−1)·1 s`,
not to the full displacement. C37 showed that these rules match real reactions to the
unexpected part of attacker movement (β ≈ 1.0–1.3), while the full displacement double counts
motion the defenders were already making (β 0.02–0.47). The innovation therefore attributes to
`a` only the response C37 supports; this is conservative. The full displacement is reported as
a sensitivity (an upper bound). If `v(t−1)` is unknown the policy changes are NaN.

`constant_velocity` is valued within its own background, because `V(observed) − V(cv)` would
credit each attacker in turn with every defender's unexpected movement, including reactions to
the ball and to teammates.

**Samples.** Attempted passes (rules v2), 150 per Metrica match and 100 per SkillCorner match,
seed 0. Off-ball outfield attackers with an earlier position; on SkillCorner the primary set
keeps attackers *detected* at `t` and `t − 1 s` (all attackers are a sensitivity). Complete
cases only (exclusions counted).

**Null.** A rotation placebo keeps each attacker's displacement and innovation size and
randomises the direction (defenders untouched).

## Hypotheses and decision rules
95% possession-clustered bootstrap CIs (1,000); SkillCorner also has a two-stage match-level CI.

* **H-R1 (response adjustment).** Under `marker_follows` and `lane_cover`, mean(change − static
  change) ≠ 0. Expected sign positive: removing the response puts the defender back near the
  attacker's earlier position or lane, crediting the run with the space it dragged defenders
  out of.
* **H-R2 (assumption dependence).** Mean between-assumption range ≥ 25% of mean |static change|
  (lower CI bound).
* **H-R3 (null).** Real − placebo change > 0 under each assumption.
* **Ranking agreement (descriptive).** Within-frame Spearman and top-1 agreement with `static`;
  "assumption-robust" if every top-1 agreement ≥ 0.8.
* **O1 (receiver).** Top-1 accuracy of the `own` change for the actual receiver of completed
  passes; response-adjusted better iff the CI of the difference from static is > 0. Expected:
  not better.
* **O2 (possession progress).** Spearman correlation of the frame's summed `forward` change with
  the largest forward ball movement over the next 10 s of the possession (observed ball frames
  only); response-adjusted better iff the CI of the correlation difference is > 0.

## What this is not
Model-based comparisons of modelled opportunity. The mixed states never happened and the option
model extrapolates to them. The response rules are assumptions with a necessary observational
check only. Not causal, not "player value".

## Results
_To be added after the pre-registered runs._
