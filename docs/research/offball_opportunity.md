# Off-ball opportunity under reactive defenders (session 4)

Status: **complete (negative for the response adjustment)**. Everything up to "Results" was
written and committed (66c16db) before the study was run on full data; only a 2-match pilot was
run, to check code paths and runtime, and its numbers are not used. Both runs are from that clean
revision. Code: `src/footymind/ghost/offball_opportunity.py` (definitions),
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
Samples (complete cases; no row had an unknown innovation):

| | passes sampled | frames used | attacker rows (primary) | clusters | matches |
|---|---|---|---|---|---|
| Metrica | 450 | 349 | 3,149 | 248 | 3 |
| SkillCorner, detected at `t` and `t − 1 s` | 2,000 | 1,677 | 10,042 (of 15,704) | 1,163 | 20 |

Most skipped frames are passes whose possession, per the possession table, still belonged to
the other team 0.2 s earlier (86 of the 100 skipped on Metrica; most of the rest had no observed
ball), so the first pass after a regain is under-represented.

### Modelled opportunity change, primary measure (best progressive option)
Mean change, possession-clustered 95% CI. SkillCorner match-level CIs in the text below.

| assumption | Metrica mean | − static | real − placebo | SkillCorner mean | − static | real − placebo |
|---|---|---|---|---|---|---|
| static | +0.0042 [+0.0025, +0.0059] | | +0.0042 [+0.0022, +0.0063] | +0.0026 [+0.0018, +0.0035] | | +0.0037 [+0.0026, +0.0049] |
| constant velocity | +0.0046 [+0.0028, +0.0064] | +0.0005 [−0.0001, +0.0011] | +0.0043 [+0.0022, +0.0064] | +0.0030 [+0.0021, +0.0039] | +0.0003 [+0.0000, +0.0007] | +0.0038 [+0.0027, +0.0050] |
| marker_follows (innovation) | +0.0040 [+0.0023, +0.0058] | −0.0002 [−0.0004, +0.0000] | +0.0040 [+0.0021, +0.0061] | +0.0026 [+0.0018, +0.0035] | −0.0000 [−0.0001, +0.0001] | +0.0037 [+0.0026, +0.0049] |
| lane_cover (innovation) | +0.0040 [+0.0024, +0.0058] | −0.0001 [−0.0003, +0.0000] | +0.0040 [+0.0019, +0.0060] | +0.0026 [+0.0017, +0.0035] | −0.0001 [−0.0001, +0.0000] | +0.0036 [+0.0025, +0.0048] |
| marker_follows (full, sensitivity) | +0.0036 [+0.0019, +0.0054] | −0.0005 [−0.0010, +0.0000] | | +0.0021 [+0.0012, +0.0030] | −0.0005 [−0.0008, −0.0003] | |
| lane_cover (full, sensitivity) | +0.0038 [+0.0020, +0.0055] | −0.0004 [−0.0007, −0.0001] | | +0.0022 [+0.0013, +0.0030] | −0.0005 [−0.0006, −0.0003] | |

Distribution (static; the others are alike): the median change is 0 on both providers and only
19% (Metrica) / 23% (SkillCorner) of attackers change the best progressive option at all
(p90 +0.0000 / +0.0005). Runners (≥ 3 m in the second) carry most of it: static +0.0121
[+0.0063, +0.0186] (810 rows) and +0.0076 [+0.0049, +0.0102] (2,844 rows).

SkillCorner match-level (two-stage) CIs: static +0.0026 [+0.0015, +0.0038]; real − placebo
+0.0037 [+0.0021, +0.0055]; marker_follows − static −0.0000 [−0.0001, +0.0001]; lane_cover −
static −0.0001 [−0.0001, +0.0000]. All attackers (extrapolated runs included, 15,704 rows):
static +0.0021 [+0.0014, +0.0027], same conclusions.

Secondary measures. Summed completion probability: Metrica marker_follows − static −0.0004
[−0.0015, +0.0007], lane_cover −0.0011 [−0.0018, −0.0003]; SkillCorner +0.0009 [+0.0004,
+0.0015] and −0.0005 [−0.0008, −0.0002] (static means +0.0053 / +0.0075). Own option: within
±0.0006 of static for both rules on both providers. With the full attribution the rules remove
up to 0.005 (e.g. lane_cover, own option, −0.0052 Metrica / −0.0048 SkillCorner), more than the
static mean change itself.

### Sensitivity and ranking agreement (primary measure)

| | Metrica | SkillCorner |
|---|---|---|
| mean range over the 4 assumptions / mean \|static\| (H-R2) | **0.62** [0.51, 0.76] | **0.65** [0.58, 0.72] (match-level [0.57, 0.74]) |
| same, static and the two rules only | 0.16 [0.12, 0.21] | 0.20 [0.17, 0.23] |
| attackers whose sign differs between assumptions | 12.8% | 20.1% |
| within-frame Spearman with static: CV / marker / lane | 0.72 / 0.57 / 0.67 | 0.75 / 0.53 / 0.66 |
| top-1 attacker agreement with static: CV / marker / lane | 0.76 / 0.84 / 0.88 | 0.80 / 0.71 / 0.77 |
| own-option top-1 agreement: CV / marker / lane | 0.59 / 0.91 / 0.96 | 0.67 / 0.89 / 0.96 |

On the primary measure many attackers tie at 0, so top-1 agreement there is partly trivial
(argmax of ties); the own-option measure has no ties.

### Outcome checks

| | Metrica | SkillCorner |
|---|---|---|
| O1 frames (completed passes, receiver analysed) / chance top-1 | 273 / 0.111 | 1,115 / 0.176 |
| O1 top-1, static | 0.092 [0.061, 0.127] | 0.178 [0.156, 0.199] |
| O1 marker_follows − static | +0.000 [−0.014, +0.015] | −0.013 [−0.024, −0.002] |
| O1 lane_cover − static | −0.004 [−0.017, +0.008] | −0.004 [−0.012, +0.002] |
| O1 constant velocity − static | +0.018 [−0.014, +0.051] | −0.022 [−0.038, −0.004] |
| O2 Spearman with possession progress, static | −0.034 [−0.140, +0.087] | +0.056 [+0.008, +0.102] |
| O2 marker_follows − static | +0.008 [−0.032, +0.049] | −0.008 [−0.031, +0.014] |
| O2 lane_cover − static | −0.001 [−0.032, +0.027] | +0.004 [−0.011, +0.018] |

### Decisions against the pre-registered rules

* **H-R1 not supported on either provider.** With the C37-supported innovation attribution,
  putting the rule responses back changes the mean change on the primary measure by at most
  0.0002, and the CIs include 0. The expected positive "drag credit" is not seen. Only the full
  attribution, which C37 says double counts, moves it (about −0.0005, i.e. 12–20% of the static
  mean, in the *negative* direction: crediting the defenders' co-movement to the run lowers the
  run's value).
* **H-R2 supported, but not because of the responses.** The between-assumption range is about
  0.6 of the static change on both providers, above the 0.25 bar. Almost all of it comes from the
  constant-velocity background (what defenders did for *other* reasons in that second). Among
  the assumptions that keep the observed state as the factual (static and the two rules) the
  ratio is 0.16 / 0.20, below the bar.
* **H-R3 supported on both providers.** Real runs raise modelled opportunity more than
  randomly directed displacements of the same size, by +0.0042 (Metrica) and +0.0037
  (SkillCorner, match-level [+0.0021, +0.0055]), under every assumption. The response
  adjustment itself is *not* specific to real runs (adjustment real − placebo within ±0.0003).
* **Ranking not assumption-robust** (top-1 agreement below 0.8 for at least one assumption on
  each provider).
* **O1, O2: response-adjusted is not better than static on either provider.** On SkillCorner it
  is slightly worse for receiver identification (marker_follows −0.013, constant velocity
  −0.022). Static itself is at chance for the receiver (as `run_value` was), and the only
  non-null progress association is static on SkillCorner (+0.056, small).

## Interpretation
* Off-ball movement in the second before a pass is purposeful in the model's eyes: real runs
  beat randomly directed ones (H-R3). The average effect on the best progressive option is
  small (+0.003 to +0.004 probability), concentrated in a minority of runners, and zero for most
  attackers.
* The defender-response question has a clear answer under the responses we can support: the
  innovation-driven `marker_follows` and `lane_cover` responses are too small, relative to the
  option model's sensitivity, to change the credit on average (H-R1). They are 0.5 × the
  unexpected part of the run for one marker, or a kinematically clipped lane shift for one
  defender, and the best-option measure rarely reacts to one defender moving under a metre.
* What the credit does depend on is the *background* convention: whether defenders are held
  where they ended up or where they would have been had nobody reacted to anything (H-R2,
  ranking agreement 0.59–0.80). That is the same convention dependence the movement
  decomposition found (C31/C32), now shown per attacker.
* None of the variants identifies the receiver or tracks possession progress better than the
  static convention. So there is no evidence here that a response-adjusted off-ball credit is a
  better quantity, only that it is a different one when the attribution is generous.

## Limitations
* Model-based throughout: mixed states never happened, the option model extrapolates to them,
  and its max-over-options measure is insensitive to small single-defender moves.
* Single-attacker interventions: real runs happen together and responses to several runs are not
  additive. The rules are a priori and passed only a necessary observational check (C37);
  `marker_follows` has no kinematic limit; `lane_cover` moves one defender.
* The constant-velocity background uses each defender's causal velocity 1 s earlier; defenders
  with no earlier state stay observed. The rotation placebo can place an earlier position off
  the pitch.
* Selection: frames whose possession changed within the 0.2 s guard are skipped (about 19% on
  Metrica); on SkillCorner the primary set is on-camera attackers (64% of rows).
* Metrica has 3 matches; its CIs reflect within-match sampling only.

## Recommendation
Do not expose a per-attacker response-adjusted "off-ball value" in the API or UI: it fails both
outcome checks and its ranking is not assumption-robust. If anything is shown, it should be the
negative finding itself, as experimental context next to the response-policy selector: "under
the response rules that passed C37, accounting for the defenders' reaction changes this run's
modelled credit by almost nothing; the credit mainly depends on whether defenders are held where
they ended up".
