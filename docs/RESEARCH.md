# Research log

Living record of research questions, experiment designs, results (including negative
ones) and prior art. Quotable numbers must also appear in `CLAIMS_LEDGER.md`.

## RQ1: Does tracking-derived spatial context improve attempted-pass outcome prediction over event-only information?

### Design (fixed before results; ADR-0005)
* **Data:** Metrica Sample Games 1–2 (anonymised; 25 Hz optical tracking + synchronised
  events). 2,236 attempted passes, 78.8% completed (1,763 `PASS`, 387 `BALL LOST`, 86 `BALL OUT`).
* **Label:** completed (1) vs. failed (0) attempted pass; rules in `DATA_SOURCES.md`.
* **Information cutoff:** decision-frame positions; causal velocities; release
  direction/speed within ≤ 0.12 s, truncated before the outcome frame; prior events.
  End locations, durations, receiver ids and outcome subtypes are excluded.
* **Model families:**
  * A (event baseline): ball location, distance/angle to goal, release direction and speed,
    body part/cross/goal-kick flags, restart flag, previous event type, pass index and time
    in possession, period, match minute.
  * B (spatial): A + pressure on the passer, candidate-receiver geometry (inferred from the
    release direction), PPCF at the candidate, lane clearance/margins, and defenders in the
    release cone.
  * C (interaction): B + second and third candidate options, best available option, marking
    distance of the candidate, defensive line depth and compactness, and team width/depth.
  * Physics: the uncalibrated PPCF × lane-clear estimate for the inferred receiver, raw and
    Platt-recalibrated.
  * Learners: L2 logistic regression and shallow gradient boosting, both with a-priori
    hyperparameters.
* **Validation:** leave-one-match-out (2 folds). Pooled out-of-fold metrics. Paired
  bootstrap over possessions (2,000 resamples, stratified by match) for differences.

### Results (`experiments/results/pass_outcome_v1*`)

Pooled out-of-fold, n = 2,236:

| model | log loss | Brier | AUC | ECE |
|---|---|---|---|---|
| base rate | 0.524 | 0.170 | 0.455 | 0.060 |
| A logistic | 0.438 | 0.142 | 0.777 | 0.036 |
| A GBM | 0.423 | 0.137 | 0.797 | 0.026 |
| physics (raw, uncalibrated) | 0.784 | 0.210 | 0.744 | 0.184 |
| physics (Platt) | 0.464 | 0.150 | 0.732 | 0.041 |
| B logistic | 0.369 | 0.117 | 0.863 | 0.029 |
| B GBM | 0.371 | 0.120 | 0.858 | 0.037 |
| C logistic | 0.371 | 0.117 | 0.861 | 0.028 |
| C GBM | 0.365 | 0.118 | 0.861 | 0.036 |

Paired differences (b − a, 95% CI):

| comparison | Δ log loss | Δ AUC |
|---|---|---|
| A logistic → B logistic | −0.069 [−0.089, −0.050] | +0.085 [+0.068, +0.105] |
| A GBM → B GBM | −0.052 [−0.072, −0.033] | +0.061 [+0.043, +0.080] |
| B GBM → C GBM | −0.007 [−0.016, +0.003] | +0.004 [−0.003, +0.010] |
| B logistic → C logistic | +0.002 [−0.006, +0.010] | −0.001 [−0.008, +0.005] |
| A logistic → physics (Platt) | +0.026 [+0.009, +0.043] (worse) | −0.046 [−0.074, −0.017] |

Ablations (logistic regression, `pass_outcome_v1_ablation`):

| comparison | Δ log loss | Δ AUC |
|---|---|---|
| A → A + pressure on passer | −0.007 [−0.013, −0.001] | +0.008 [+0.001, +0.017] |
| A → A + defenders in the release path | −0.008 [−0.017, +0.000] | +0.022 [+0.012, +0.033] |
| A → A + **defender-only** (pressure + path; no teammate info) | **−0.015 [−0.025, −0.005]** | +0.028 [+0.014, +0.042] |
| A → A + teammate geometry (candidate distance/alignment/progress) | −0.058 [−0.074, −0.044] | +0.062 [+0.047, +0.078] |
| A + teammate geometry → B (adds defender context) | −0.010 [−0.024, +0.003] | +0.023 [+0.010, +0.036] |

Sensitivity analyses reproduced the direction and approximate size of every effect:
* excluding `BALL OUT` (n = 2,150): A→B −0.056 [−0.076, −0.038]; defender-only −0.024 [−0.035, −0.012];
* excluding the 22 attempts whose release window was truncated (n = 2,214): A→B −0.062
  [−0.083, −0.043]; defender-only −0.012 [−0.022, −0.001].

The receiver inferred from the release direction matched the true receiver on 78.6% of
completed passes (n = 1,763).

### RQ1 v2: three matches (`pass_outcome_v2_3match`, revision dffc3e8)

Metrica Game 3 was added through the EPTS adapter. Its JSON annotation adds `CARRY`
events, which are excluded from history features. That makes its fold a **transfer test
to a different annotation scheme**. There are 3,633 attempted passes (79.2% completed).
The receiver inference is 76.1% top-1.

| model | pooled log loss | AUC | ECE | G1 fold | G2 fold | G3 fold (transfer) |
|---|---|---|---|---|---|---|
| A logistic | 0.429 | 0.784 | 0.030 | 0.469 | 0.402 | 0.422 |
| A GBM | 0.400 | 0.817 | 0.016 | 0.438 | 0.371 | 0.397 |
| B logistic | 0.358 | 0.866 | 0.026 | 0.393 | 0.320 | 0.364 |
| B GBM | 0.350 | 0.868 | 0.019 | 0.377 | 0.322 | 0.354 |
| C GBM | 0.349 | 0.869 | 0.017 | 0.376 | 0.317 | 0.355 |
| physics (raw) | 0.772 | 0.751 | 0.174 | 0.823 | 0.749 | 0.752 |

| comparison | Δ log loss [95% CI] | Δ AUC [95% CI] |
|---|---|---|
| A LR → B LR | −0.071 [−0.085, −0.057] | +0.082 [+0.068, +0.096] |
| A GBM → B GBM | −0.050 [−0.062, −0.037] | +0.051 [+0.038, +0.064] |
| B GBM → C GBM | −0.002 [−0.006, +0.003] | +0.001 [−0.003, +0.005] |
| A LR → A + defender-only | −0.014 [−0.021, −0.008] | +0.022 [+0.013, +0.030] |
| A LR → A + teammate geometry | −0.047 [−0.059, −0.035] | +0.054 [+0.041, +0.066] |
| A LR → physics (Platt) | +0.027 [+0.013, +0.041] (worse) | −0.036 [−0.058, −0.014] |

Annotation sensitivity (`pass_outcome_v2_sens_no_after_carry`) drops all attempts
immediately preceded by the same player's CARRY, in both classes (n = 2,876). It gives
A→B −0.072 [−0.088, −0.056] and defender-only −0.011 [−0.019, −0.004].

The v2 conclusions match v1. The spatial gain holds in every fold, including the transfer
fold. Defender-only information gives a small, consistently positive gain. Interaction
features add nothing measurable, and the raw physics estimate stays miscalibrated.

### Interpretation
1. **Spatial context helps, robustly, on held-out matches.** It reduces log loss by about
   16% relative to the event baseline (0.438 → 0.369) and improves discrimination
   (AUC 0.78 → 0.86).
2. **Most of the gain comes from where teammates are relative to the pass direction**, not
   from defensive pressure. Defender-only information gives a smaller but statistically
   detectable improvement (about 0.015 log loss). This is the honest current answer to "does
   defensive positioning carry measurable signal?": yes, but modestly, at this data scale.
3. **Interaction features did not help** beyond B (C ≈ B). This does not justify graph
   neural networks or other expressive models (ADR-0003).
4. **The literature-parameter physics estimate is poorly calibrated** for observed passes
   (ECE 0.18; overconfident). Even recalibrated, it is worse than the event baseline. Likely
   causes: (i) the receiver is inferred, not known (21% wrong); (ii) the target is the
   receiver's current position, with no lead passes; (iii) the uniform reaction time makes
   short-range lane interceptions unlikely (see `spatial/lanes.py`); (iv) the parameters
   are not fitted to these data. **The scenario engine's p_success must therefore be
   presented as an uncalibrated model estimate**, which the API and UI already do.

### Limitations
* Two matches from one anonymised provider sample. Between-match and between-competition
  variability cannot be estimated, so the CIs are within-match sampling intervals.
* Retrospective observational data. Players chose these passes. Results describe
  predictability of chosen actions, not the success of alternative passes (selection bias).
* Failed-pass labels depend on Metrica's annotation and our mapping rules.
* The candidate-receiver inference may itself encode outcome-correlated execution error
  (a mishit pass points at no one). That is a property of the executed action, which is
  observable at release, but it means "teammate geometry" partly measures execution
  quality, not only tactical availability.

### Relation to prior work
Predicting pass completion from tracking is established. Power et al. (2017) modelled
pass risk and reward from tracking, and Anzer & Bauer (2022) built expected-pass models from
spatio-temporal data. Our result is **not a novel finding**. It is an open, reproducible,
leakage-audited replication on public data. Its additional contributions are the explicit
decomposition into teammate geometry vs. defender-only signal and the calibration audit of
a literature physics model.

## Option model (serving model for pass options)

**Purpose.** The scenario engine needs a success probability for *any* receiver, including
options that were never chosen and hypothetical states. Model B cannot provide one, because
its strongest feature (the release direction's alignment with a teammate) is undefined for
unchosen options. The option model is a logistic regression on per-option features computable
for any target: ball and target location, distance, forward progress, PPCF at the target, lane
clearance, margin and perpendicular distance, defenders near the lane, pressure on the passer,
the target's nearest-defender distance, pass forwardness and receiver movement along the pass.
Training uses the inferred candidate receiver. The features come from the same function used in
serving (`spatial/option_features.py`), and a test asserts equality on real frames.

**Iteration log (honest).** v0 had 15 features: LOMO log loss 0.428, AUC 0.780, ECE 0.019.
After that evaluation, 3 domain-motivated features were added once (marking distance at the
target, forwardness, receiver movement along the pass). v1: **LOMO log loss 0.422, AUC 0.789,
ECE 0.017** (3 matches, 3,633 passes). No further iterations were made.

**Interpretation.** The option model is well calibrated, and it is better than the physics
estimate, raw (0.772) or recalibrated (0.456). Its discrimination is similar to the
event-only baseline and below Model B (0.358). This is expected, because B exploits
the executed release direction. It is the best *option-agnostic* estimate currently available,
and the API reports it next to the physics heuristic with its held-out metrics and the
selection-bias caveat.

## Prior art and references
* Spearman, W., Basye, A., Dick, G., Hotovy, R., Pop, P. (2017). *Physics-Based Modeling of Pass Probabilities in Soccer.* MIT Sloan Sports Analytics Conference.
* Spearman, W. (2018). *Beyond Expected Goals.* MIT Sloan SAC. (Pitch control; off-ball scoring opportunity.)
* Fernández, J., Bornn, L. (2018). *Wide Open Spaces: A statistical technique for measuring space creation in professional soccer.* MIT Sloan SAC.
* Fernández, J., Bornn, L., Cervone, D. (2021). *A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions.* Machine Learning 110. (EPV; SSAC 2019 version.)
* Power, P., Ruiz, H., Wei, X., Lucey, P. (2017). *Not all passes are created equal: Objectively measuring the risk and reward of passes in soccer from tracking data.* KDD.
* Anzer, G., Bauer, P. (2022). *Expected passes: Determining the difficulty of a pass in football (soccer) using spatio-temporal data.* Data Mining and Knowledge Discovery 36.
* Decroos, T., Bransen, L., Van Haaren, J., Davis, J. (2019). *Actions Speak Louder than Goals: Valuing Player Actions in Soccer.* KDD. (VAEP; `socceraction`.)
* Singh, K. (2019). *Introducing Expected Threat (xT).* (Blog post.)
* Le, H. M., Carr, P., Yue, Y., Lucey, P. (2017). *Data-Driven Ghosting using Deep Imitation Learning.* MIT Sloan SAC. (Learned defensive response, relevant to future response policies.)
* Wang, Z., Veličković, P., Hennes, D., et al. (2024). *TacticAI: an AI assistant for football tactics.* Nature Communications 15.
* Shaw, L. (2020). *Friends of Tracking: LaurieOnTracking* (reference pitch-control parameterisation).

### What is original here vs. adapted
* Adapted: the PPCF model and parameters (Spearman; Shaw), the pass-risk modelling idea (Power; Anzer & Bauer).
* Engineering and methodology contributions: provenance-labelled counterfactual engine with
  reachability constraints and parameter-ensemble sign-agreement; leakage-audited pass
  dataset on public data; teammate-geometry vs. defender-only decomposition.
* Open research directions (not yet done): data-calibrated PPCF; Defensive Shadow
  (displacement sensitivity of option value per defender); off-ball valuation (Ghost Value).

## Log
* **2026-11-13**: Option model v1 trained (LOMO log loss 0.422, AUC 0.789, ECE 0.017). It is now the default value model in scenarios.
* **2026-11-13**: Added the Metrica Game 3 EPTS adapter (direction cross-checked against the
  provider metadata). RQ1 v2 with 3 folds confirms v1 in every fold.
* **2026-11-13**: Legacy audit (v1 predictor had no generalization evidence). Metrica
  ingestion validated (sync 0.08 m). RQ1 v1 run, plus ablation and two sensitivity analyses.
  Found and documented two leakage traps: failed-pass end location = loss location, and
  centred velocity filters read future frames.
