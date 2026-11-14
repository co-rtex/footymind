# FootyMind: The Invisible Game. Technical research report

_Version 1.0 release candidate. Every number below is traceable to a row of
[`CLAIMS_LEDGER.md`](CLAIMS_LEDGER.md) (the `Cxx` labels), which points to a committed
`experiments/results/*/results.json` or a test. The chronological lab notebook, with full
tables, superseded analyses and review responses, is [`RESEARCH.md`](RESEARCH.md)._

## Abstract

Event data records what happened to the ball: passes, shots, tackles. It does not record
the space, pressure and passing options that shaped each decision. FootyMind is an
open-source lab that reconstructs that "invisible game" from player-tracking data from two
providers. It studies how much it adds to the analysis of decisions, using pre-registered,
leakage-audited, leave-one-match-out experiments. Without any information from the executed
kick, tracking-derived spatial context improves held-out prediction of attempted-pass
outcomes over event-only context on both optical tracking (Metrica, 3 matches:
Δ log loss −0.026 [−0.034, −0.018]) and broadcast tracking (SkillCorner, 20 A-League
matches: −0.021 [−0.025, −0.018], better in 20/20 held-out matches). The gain survives
reading every feature 0.2 s before the release. The last second of movement adds a further,
smaller gain on both providers (−0.014 and −0.009). Several more ambitious ideas did not
survive evaluation, and they are reported as negative results:
* a literature physics model of pitch control is structurally miscalibrated;
* a defender "shadow" metric is no better than the nearest defender;
* a simple off-ball "run value" ranks receivers near chance, far below simple baselines;
* crediting a run with the defenders' reaction to it changes its modelled value by almost
  nothing, under every response the data support; the credit depends on the background
  convention instead.

The counterfactual simulator ("Possession Multiverse") is therefore presented as a
model-based comparison with explicit, partly validated opponent-response assumptions, never
as causal evidence.

## 1. Motivation

A pass completion rate, a passing network or an expected-goals number summarises outcomes.
None of them sees why a pass was on: how close the nearest defender was and how fast they
were closing, whether a lane was open, which teammate had space. Tracking data records every
player's position many times a second. That information is in principle available, but using
it credibly raises three problems:

1. **Leakage.** Many features that "help" a pass-outcome model are measured during or after
   the action: the end location, the receiver, the kick itself. A model that sees them
   learns execution, not decision context.
2. **Provider dependence.** Optical and broadcast tracking differ in frame rate, visibility
   and event annotation. A result that holds on one provider may be an artefact of it.
3. **Counterfactual over-reach.** It is tempting to move a defender and read the change in a
   model score as "what would have happened". Without a validated model of how opponents
   respond, that is a statement about a model, not about football.

FootyMind is built around these three problems. Every decision-time feature is causal and
read before the kick. Every experiment is run per provider and never pooled. Every simulated
state is labelled as hypothetical or simulated, with its assumptions shown.

## 2. Data

| provider | matches | sampling | tracking | events | licence |
|---|---|---|---|---|---|
| Metrica Sports sample data | 3 (anonymised) | 25 Hz | optical, all players observed | Metrica events (CSV and EPTS) | public sample, no licence file |
| SkillCorner open data | 20 (A-League 2024/25) | 10 fps | broadcast; off-camera players and ball are provider extrapolations | SkillCorner dynamic events (adapter v2) | MIT, credit requested |

Canonical representation (ADR-0002): metres, origin at the centre spot, the attacking
direction inferred per period from goalkeeper positions and cross-checked against
provider-declared directions. Ingestion fails on any contradiction (C2, C13). Every position
carries a provenance label:

| label | meaning |
|---|---|
| observed | measured |
| derived | computed from observations |
| estimated | provider extrapolation or model output |
| hypothetical | a user's edit |
| simulated | a response policy's move |

Unobserved values are NaN and are never filled in.

**Data quality** (C1, C30, C38).
* Event–tracking synchronisation is tight on Metrica: the median passer–ball distance at the
  pass start is 0.084–0.088 m.
* On SkillCorner, 66.6% of on-pitch positions at pass starts are camera-detected (87.4% within
  25 m of the ball). The ball's own detection flag is kept, and an extrapolated ball is
  labelled estimated.
* **On SkillCorner, the event start frame often already contains the kick.** The ball is
  already moving away from the passer in 62.7% of passes there, against 1.8% on Metrica. By
  0.2 s earlier the share has fallen to the broadcast noise floor (C38). This drives the
  guard-frame rule in §4.

## 3. Research questions

* **RQ1.** Does tracking-derived spatial context improve prediction of attempted-pass outcomes
  beyond event-only context, before the pass is played? Does the answer replicate across
  providers?
* **RQ2.** Does the recent evolution of the situation (time on the ball, the last second of
  movement) add information beyond a static snapshot?
* **RQ3.** Can physics-based models (pitch control, lane interception) supply calibrated pass
  probabilities?
* **RQ4.** How reliable are the modelled tactical interventions? Specifically: the opponent
  responses, the physical movement constraints, and the off-ball opportunity estimates.
* **RQ5.** Do proposed "invisible" player metrics (Defensive Shadow, run value) measure
  anything beyond simple baselines?

## 4. Methods

**Task (ADR-0005).** The unit is an attempted pass, and the label is whether it was
completed. Failed passes are interceptions, possession losses and passes out of play, under a
symmetric subtype allow-list (rules v2). The intended target of a failed pass is unobserved,
so it is never used as a label.

**Feature families.**
* *Event-only decision-time* (`Adt`): ball location, distance and angle to goal, restart flag,
  previous event type, pass index and time in the possession, and match clock.
* *Spatial decision-time* (`Bdt`): `Adt` plus pressure on the passer (nearest-defender distance,
  closing speed, defenders within 5/10 m), passer speed, defensive shape (defenders goal-side
  of the ball, line depth, compactness), attacking width and depth, and the best and number of
  physically plausible options.
* *Temporal* (RQ2): time-on-ball from tracking (censored when an unobserved ball interrupts it)
  and 1 s movement history (change in nearest-defender distance, ball travel, forward movement
  of the passer and both team centroids, change in defensive shape and in open lanes).
* *Action-conditioned* (`A`, `B`, `C`): the release direction and speed over at most 0.12 s,
  truncated before the outcome frame. These answer a different question ("given the kick")
  and are reported separately.

**Leakage controls.**
* Velocities are causal (a backward 0.2 s window), and every temporal window ends at the
  decision frame. A regression test corrupts every later frame and checks that no feature
  changes.
* Event end locations, durations, receiver identities and outcome subtypes are never features.
* **Guard frame (ADR-0007).** Decision-time claims for SkillCorner use features read 0.2 s
  before the event start frame, because the start frame can already contain the kick. Metrica
  results are reported at both frames.
* Time-related constants are defined in seconds and converted per provider. Frame-count
  windows were found and fixed in two places: the shadow interception-recovery window
  (5 s instead of 2 s at 10 fps) and the event-carrier tolerance.

**Evaluation.**
* Leave-one-match-out (LOMO); a test match never contributes training rows.
* Hyperparameters fixed a priori: logistic regression with C = 1 and median imputation, and a
  shallow gradient-boosting model. No tuning on test folds.
* Uncertainty: a paired bootstrap over possessions within matches. For SkillCorner's 20
  matches, a two-stage bootstrap (matches, then possessions) is also reported, along with the
  number of held-out matches in which the richer model wins.
* Comparisons are pre-registered in experiment configs and in `RESEARCH.md` before the
  results are seen. Amendments made after an independent review are dated and justified.
* Providers are never pooled, and each has its own option model.

**Independent reviews.** Two methodology reviews were run in session 3. They found:
* censoring and kick-leakage problems in the temporal features;
* that the off-ball interaction statistic was mostly model non-additivity;
* double counting in the first response-policy check;
* a served similar-moments statistic that differed from the validated one.

All were fixed or reported as revisions (see the "revised" ledger rows). Two more independent
reviews in session 4 covered release methodology and engineering. They led to:
* a size-matched, paired placebo for the rule-policy check (v4);
* guarded held-out metrics for the serving option model;
* corrected wording on what each response model is validated for;
* period-safe SkillCorner event ends;
* cache freshness checks.

The session-3 reviews prompted the guard frame itself.

## 5. Results

### 5.1 RQ1: spatial context, replicated across providers

| | Metrica (3 matches) | SkillCorner (20 matches) |
|---|---|---|
| decision-time, event-only → + spatial | **−0.026** [−0.034, −0.018], AUC 0.648 → 0.721, 3/3 matches (C20) | **−0.021** [−0.025, −0.018], 20/20 matches (C28) |
| same, every feature 0.2 s before release | −0.026 [−0.034, −0.018] (C34) | −0.021 [−0.025, −0.018] (C35) |
| action-conditioned (kick known) | −0.071 (C20) | −0.046 (C28) |
| interaction features beyond spatial | none (C12) | −0.004 (C28) |

The decision-time gain is the primary, replicated result. It is not an artefact of kick
information, because it is unchanged at the guard frame. On SkillCorner the leading spatial
drivers are the number of defenders goal-side of the ball and attacking width (C26). The
action-conditioned gain is larger and provider-dependent, because much of it reflects where
the ball was actually played, which is partly execution quality.

**Serving model.** A per-provider logistic option model over per-teammate features, including
lead-pass targets, gives held-out (LOMO) log loss 0.414, AUC 0.796, ECE 0.019 on Metrica (C21)
and 0.354 / 0.811 / 0.024 on SkillCorner (C24). With every input read at the 0.2 s guard, the
figures are 0.419 / 0.790 / 0.028 and 0.354 / 0.810 / 0.021 (C71), so quality does not depend
on kick information. These are the probabilities shown in the
product. They describe the passes players chose to attempt, so estimates for options nobody
chose are extrapolations.

### 5.2 RQ2: the last second matters

Comparison of Δ log loss when 1 s of movement history is added to static spatial context
plus time-on-ball (H-T2), at the guard frame:

| | Metrica | SkillCorner |
|---|---|---|
| H-T2 | **−0.014** [−0.021, −0.007] (C33) | **−0.009** [−0.011, −0.007], 20/20 matches, match-level CI [−0.012, −0.006] (C35) |
| time-on-ball over the event baseline (H-T1) | −0.002 [−0.005, +0.001] (C34) | −0.002 [−0.003, −0.001] (C35) |

The movement gain is mainly in discrimination (AUC rises), not calibration. At the guard frame
it holds with a gradient-boosting model (Metrica −0.013). At the release frame it also holds
with header/cross annotations added to both sides (−0.013). On Metrica it shrinks when read 0.4 s before the release,
consistent with information ageing. On SkillCorner, the guard removes about half of the
release-frame time-on-ball gain and about a fifth of the movement gain, but none of the
spatial gain. Kick information had inflated only the temporal features.

### 5.3 RQ3: physics models are not calibrated probabilities

The literature pitch-control × lane-interception estimate is badly miscalibrated
(ECE 0.18). After Platt scaling it is worse than the event-only baseline (C7). Fitting its
parameters per fold improves calibration, but drives them to physically implausible bounds and
lowers AUC, and it still loses to the learned option model (C15). The physics model is kept for
geometry (control surfaces, lane margins). It is not used for probabilities.

### 5.4 Similar moments

For a pass decision, the completion rate among the 50 most similar decisions from *other*
matches of the same provider is computed on the 0.2 s guard-frame situation and shrunk toward
the base rate. It beats the base rate held-out on both providers (−0.057 / −0.051 log loss).
It matches the logistic model on SkillCorner and is slightly worse on Metrica (C36). The
product shows this validated estimate, separately from the short list of example neighbours.

### 5.5 RQ4: counterfactual simulation

**Opponent responses** (ADR-0006).
* A ridge model of short-horizon defender displacement forecasts real defender movement
  better than constant velocity on both providers (C40, C42). Under hypothetical interventions
  it is nearly inert, moving defenders by centimetres (C44). It is a good forecaster but not an
  intervention model.
* Four interpretable rules were tested observationally (C37; under envelope v2, C64; paired against size-matched placebos, C70), by treating the attackers'
  *unexpected* real movement (beyond their own constant velocity) as the intervention:
  * `marker_follows` and `lane_cover` reduce forecast error on both providers. The fitted size
    slope is about 1, and the gain is several times larger than a permuted placebo's.
  * `press_carrier` passed only on Metrica under the v1 envelope, and passes nowhere under v2 (C64).
  * `zonal_shift` overshoots: real blocks shift a fraction of its move (β 0.25 / 0.13, C64).
  * A first version of this check concluded that every rule was wrong. That was a
    double-counting artefact, corrected after review.

These checks are necessary conditions only. No data shows how defenders would have responded
to an intervention that never happened.

**Physical constraints.** The a-priori kinematic envelope (0.25 s reaction delay, 3.5 m/s²,
7.5 m/s) contains only 73% (Metrica) and 61% (SkillCorner) of real 1 s defender positions
(C41, C43). The reaction delay is the main cause (C57). A pre-registered envelope v2 drops the
delay, uses direction-dependent acceleration limits, and adds lag and noise margins. It covers
about 99.9% of real positions and still rejects 96–98% of 5 m shifts at 1 s (C55, C56). At
the 1.5 s scenario default, most 5 m shifts are physically reachable under any bound.

**Learning responses.** The learned position model is inert because positions barely predict
defenders' deviations from their own motion (C61). Reactions are visible only in co-movement.
An innovation model that conditions on attackers' unexpected movement beats constant velocity
and a placebo on both providers. It is served as `learned_innovation`, with the label
"observational", not causal (C62). Under envelope v2, `marker_follows` and `lane_cover` still
pass the rule check; `press_carrier` no longer does (C64).

### 5.6 RQ5: "invisible" player metrics

* **Defensive Shadow** (suppression of passing value by a defender's position) is robust to
  tracking noise (C16). It finds the interceptor at about twice chance on Metrica, but no better
  than the nearest defender (C17). On all 20 SkillCorner matches it is at chance and worse than
  nearest-defender in 18/20 (C29).
* **Run value** (an attacker's availability gain from their last second of movement, with
  defenders held fixed) shows a tiny positive association with being chosen. As a ranking it is
  near chance on both providers (Metrica top-1 0.134 against 0.100 chance; SkillCorner at
  chance) and far below current availability or speed (C22, C25).
* **Why off-ball credit is hard** (C31, C32). A Shapley decomposition of the change in team
  opportunity over the second before a pass shows two things. On average, attacking movement
  creates progressive options and defensive movement removes about as many. The interaction
  between the two is large, but a rotation placebo shows it is mostly the option model's own
  non-additivity, with a small genuine attacker–defender coupling on both providers. Per-pass credit for a run
  therefore depends on which side is held fixed, a convention rather than a fact.

### 5.7 Off-ball value under reactive defenders (priority A)

**Question.** If an attacker's run in the second before a pass dragged defenders with it, does
crediting the run with that reaction change its value? The study (pre-registered in
`docs/research/offball_opportunity.md` before any full run) places one attacker back where they
were 1 s earlier, at the 0.2 s guard frame, and values the team's best progressive option under
four defender assumptions:

* static (defenders where they ended up);
* constant velocity (where they would have been had nobody reacted to anything);
* `marker_follows` and `lane_cover` responses to the attacker's *unexpected* movement, the
  attribution the rule check supports (C70).

**Findings.**

* **Runs are purposeful in the model's eyes.** Real runs beat randomly directed ones of the
  same size on both providers: +0.0042 Metrica, +0.0037 SkillCorner (match-level
  [+0.0021, +0.0055]). The effect is tiny on average and zero for most attackers (C45).
* **Defenders' reactions barely change the credit.**
  * `marker_follows` changes it by at most 0.0002.
  * Under envelope v2, `lane_cover` lowers it by 0.0003 / 0.0001. That is non-zero, but in the
    opposite direction to the hypothesised "drag credit" (C46).
  * A post-review positive control removes a fitted response that moves 8–9 defenders. It
    changes nothing either (C47). The null is therefore not an artefact of single-defender rules.
  * It is not a strict positive control: no condition shows the measure detecting a known large
    effect.
* **The convention is what matters.** The spread across assumptions is about 0.65 of the static
  change, almost all from the constant-velocity background. The attacker ranking is not robust
  to it (C48).
* **No adjusted variant tracks outcomes better.** None identifies the receiver or tracks
  possession progress better than static (C49).

So FootyMind does not expose a response-adjusted off-ball value. The negative result is the
finding: under the responses the data support, off-ball credit is decided by the background
convention, not by defender reactions.

## 6. Negative and limiting results (summary)

| idea | outcome | claims |
|---|---|---|
| Literature pitch-control probabilities | miscalibrated; fitting drives parameters to implausible bounds | C7, C15 |
| Defensive Shadow | not better than nearest defender; at chance on broadcast data | C17, C29 |
| Run value (off-ball credit, defenders fixed) | near chance as a receiver ranking, far below simple baselines | C22, C25 |
| Off-ball interaction | mostly model non-additivity (placebo) | C31, C32 |
| Response-adjusted off-ball credit | reactions change it by ≤ 0.0003 (a fitted whole-block response too); credit depends on the background convention; no better on outcome checks | C46–C49 |
| Learned response model as an intervention model | inert under edits | C44 |
| `zonal_shift` response rule | overshoots real block movement | C37, C64 |
| `press_carrier` response rule | not supported under envelope v2 | C64 |
| Kinematic envelope v1 | missed 27–39% of real 1 s positions (reaction delay) | C41, C43, C57 |
| Interaction features (GBM) | no gain on 3 matches; −0.004 on 20 | C12, C28 |
| Time-on-ball | small; not robust on Metrica | C34, C35 |

## 7. Threats to validity

* **Few independent matches.** Three Metrica matches cannot support population-level claims.
  Their bootstrap CIs reflect within-match sampling. SkillCorner's 20 matches allow a
  match-level bootstrap, but they come from one league and season.
* **Observational data.** Every pass-outcome result is an association between pre-decision
  context and the outcome of the pass the player chose. Option-model estimates for passes not
  chosen are extrapolations (selection bias).
* **Broadcast tracking.** About a third of SkillCorner positions at pass starts are
  extrapolated (C30), and the ball is sometimes extrapolated too. Both are labelled.
  Velocities at 10 fps are noisy.
* **Event annotation.** Event start frames lag the kick on SkillCorner (C38). Failed-pass labels
  depend on documented mapping rules for each provider.
* **Model-based counterfactuals.** Scenario outputs compare model estimates under stated
  assumptions. The response policies are validated, at most, as forecasts of real play.
* **Researcher degrees of freedom.** Pre-registration, four independent reviews and the dated
  amendments reduce, but do not remove, analytic flexibility. All superseded analyses remain in
  `RESEARCH.md`.
* **Multiplicity.** Many comparisons are reported. No correction for multiple testing is applied,
  and only C18 discusses it explicitly. Borderline results should be read as fragile:
  * the run-value association (C22);
  * the decomposition real−placebo difference on Metrica (C31, lower bound +0.017);
  * Metrica time-on-ball (C34);
  * `press_carrier` (C64);
  * the off-ball secondary measures.
* **What the CIs cover.** The bootstrap CIs resample possessions (and, where stated, matches)
  for a fixed set of out-of-fold predictions. They do not include the variance from refitting
  the models.
* **Guard frame.** The 0.2 s guard was chosen after an outcome pilot and then checked against
  the pre-kick floor (C38). Where the guard-frame ball is unobserved, the guarded datasets use
  the start-frame ball. On Metrica that is 295 of 3,656 passes, 227 of them restarts where the
  ball is placed (C38). Two 2-frame rules in the pass dataset (a restart window and a minimum
  duration) are frame-count rules, so they mean 0.08 s on Metrica and 0.2 s on SkillCorner.
  They are documented, not converted, to keep committed results reproducible.
* **Envelope v2 coverage** was close to known in advance. A session-3 sensitivity grid had
  already shown that removing the reaction delay gives near-full coverage, so the coverage
  result is confirmatory. Discrimination is reported against a 5 m shift and a defender
  permutation only. Smaller shifts are mostly accepted.

## 8. Reproducibility

* `scripts/reproduce.sh --dry-run` lists every committed research artefact in dependency
  order. The script refuses to run with uncommitted source changes.
* Every results file records its `code_revision` and the input data hashes.
* The checksum-pinned data download, the deterministic ingestion and the fixed seeds make every
  number regenerable from public data. Session-4 results that predate the pipeline version 3
  fix (period-safe SkillCorner event ends) were produced on version 2 bundles. The difference
  is about one event per SkillCorner match.

## 9. Future work (not done)

* Off-ball valuation against a possession-value target, with response models identified from
  natural experiments.
* Response models that react strongly enough to matter for off-ball credit. The current
  innovation model is a co-movement forecaster. Applying it to user edits extrapolates beyond
  the real innovations it was fitted on.
* More leagues and providers, to test generalisation beyond one season.
* A tighter, pre-registered envelope evaluated on fresh frames (not pass starts), with a
  shift-size curve against a size-matched disc.

## References

Prior art is listed in [`RESEARCH.md`](RESEARCH.md#prior-art-and-references) (pitch control,
expected pass models, off-ball valuation, ghosting), with notes on how each relates to this
work.
