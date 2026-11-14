# Simulation, session 4: kinematic envelope v2 and the inert learned response

_Specialist B (counterfactual simulation). Numbers quoted here come from committed
`experiments/results/*/results.json` files named next to them; proposed ledger rows are
C55–C64._

## Part 1. Kinematic envelope v2

### Pre-registration (written and committed before any v2 coverage was computed)

**Why v1 misses real positions (hypotheses).** v1 contains 31% / 73% (Metrica) and
19% / 61% (SkillCorner) of real defender positions at 0.5 s / 1.0 s (C41, C43). Candidate
causes, each with a test:

1. *Reaction delay.* v1 holds the observed velocity for 0.25 s. A player's current
   acceleration is not observed, so the delay asserts it is zero. With an unknown
   acceleration bounded like any other, a delay does not restrict the reachable set at all.
   Test: ablation `v1_no_delay`.
2. *Acceleration bound.* v1's 3.5 m/s² is a GPS *high-intensity threshold* (Varley & Aughey
   2013 use > 2.78 m/s²), which players exceed routinely; it is not a maximum. Braking and
   turning are limited by friction and strength (peak decelerations 6–9 m/s², Harper et al.
   2019; the pitch-control literature uses 7 m/s², Spearman 2018 and its reference
   implementation). Speeding up along the run falls with speed (linear force–velocity
   profile, Samozino–Morin sprint profiling; footballers about F0 7–8 N/kg, V0 about 9 m/s).
   Test: ablations `v2_isotropic` (no force–velocity limit) and `v2_accel_3.5`.
3. *Velocity estimate.* The causal 0.2 s velocity is the mean over the last 0.2 s, so it lags
   the current velocity by about 0.1 s; it is also noisy, more so at 10 fps where the window
   is two frames. Test: noise diagnostics (below) and ablation `v2_no_lag`.
4. *Measurement error.* Positions at the start and at the horizon are noisy. Test: ablation
   `v2_no_margin`; the margin is sized from stationary-player jitter, never from coverage.
5. *The reached-point test at the horizon end.* Units: the horizon is converted with
   `round(h * fps)` (13 frames = 0.52 s at 25 Hz, 5 frames at 10 fps) and the envelope is
   evaluated at that exact time `k / fps`; `traj[:, k - 1]` is the position at frame `t + k`.
   A unit test pins the conversion.

**Envelope v2** (`counterfactual/kinematics.py`, `KinematicLimitsV2`; v1 stays bit-for-bit
and selectable):

| Parameter | v1 | v2 | Source |
|---|---|---|---|
| reaction delay | 0.25 s | none | hypothesis 1 (unknown current acceleration) |
| braking / turning bound | 3.5 m/s² | 7.0 m/s² | friction/strength; Spearman 2018, Harper et al. 2019 |
| speeding up along the run | 3.5 m/s² | 7.0 × (1 − v / 9) m/s² | linear force–velocity profile |
| speed cap | 7.5 m/s | 9.0 m/s | force–velocity intercept V0 |
| velocity lag | none | 0.1 s (half the causal window) | definition of the estimator |
| measurement margin | none | 2.45 · sqrt(2σp² + (σv T)²) | 95% radius of a 2-D Gaussian; σ from stationary jitter |
| unknown velocity | start from rest | speed-cap disc (any direction) | never turn "unknown" into 0 |
| geometry | inner polygon of constant-heading endpoints | support-function polygon of the same endpoints (convex by construction) | non-expansive projection |

**Noise (estimated independently of coverage).** `research/envelope_noise.py`: for windows of
1.0 s in which a player's least-squares line moves slower than 0.3 m/s (complete, one
period, no artefact, detected-only on broadcast tracking), σp is the per-axis RMS residual and
σv the per-axis RMS deviation of the causal 0.2 s velocity from the line slope. Pooled over
every match of the provider (the served constants): Metrica σp = 0.0173 m, σv = 0.0686 m/s;
SkillCorner σp = 0.0326 m, σv = 0.0795 m/s. Real micro-movement is included, so these are
upper bounds on pure measurement noise. The evaluation uses leave-one-match-out estimates.

**Evaluation** (`research/envelope_validation.py`, per provider, never pooled). Rows: the
`response_validation` defender rows with a complete 1.5 s trajectory. Metrics at 0.5, 1.0 and
1.5 s: coverage of real positions; rejection of real positions shifted 5 m in a uniformly
random direction; rejection of another outfield defender's real position in the same frame
(permutation); mean polygon area; cost. Possession-clustered bootstrap, 1,000 resamples,
stratified by match.

**Adoption rule.** v2 becomes the default envelope for the response policies' clipping and the
feasibility diagnostic iff, on **both** providers: coverage ≥ 0.95 at 0.5 s and 1.0 s; shift
rejection ≥ 0.90 at 0.5 s and ≥ 0.50 at 1.0 s; permutation rejection ≥ 0.90 at 0.5 s and
1.0 s; one envelope in ≤ 5 ms (median). Point estimates against thresholds; CIs reported. The
1.0 s shift threshold is lower because a sprint start covers only about 3.3 m in 1.1 s, so a
5 m offset is near the physical boundary. 1.5 s is reported, not gated. Ablations are
descriptive and cannot be adopted. A runtime pilot on one SkillCorner match is run with its
output discarded unread, to check cost and errors only.

### Results (`experiments/results/envelope_validation_v1[_skillcorner]/results.json`, rev e66719c)

Rows: Metrica 36,448 defender rows (3 matches, 1,001 possessions); SkillCorner 107,800
detected-only rows (20 matches, 4,034 possessions; 43,409 rows excluded for an extrapolated
start position and 8,555 for an extrapolated target frame). 95% CIs are possession-clustered.

| Provider | Envelope | Coverage 0.5 s | Coverage 1.0 s | Coverage 1.5 s | Area 0.5 / 1.0 / 1.5 s (m²) |
|---|---|---|---|---|---|
| Metrica | v1 | 0.311 [0.303, 0.319] | 0.725 [0.718, 0.733] | 0.858 [0.853, 0.864] | 0.05 / 3.0 / 24.6 |
| Metrica | **v2** | **0.998** [0.998, 0.999] | **0.999** [0.999, 1.000] | 1.000 [0.999, 1.000] | 5.2 / 39.2 / 142.7 |
| Metrica | disc 7 m/s (`supported`) | 0.997 | 0.998 | 0.998 | 41.7 / 154.2 / 356.2 |
| SkillCorner | v1 | 0.188 [0.184, 0.191] | 0.609 [0.604, 0.614] | 0.764 [0.760, 0.768] | 0.04 / 3.0 / 23.0 |
| SkillCorner | **v2** | **0.9997** [0.9996, 0.9998] | **0.9996** [0.9995, 0.9997] | 0.9997 | 5.0 / 40.1 / 139.2 |
| SkillCorner | disc 7 m/s (`supported`) | 0.995 | 0.996 | 0.997 | 38.5 / 154.2 / 346.9 |

| Provider | Envelope | Shift rejection 0.5 / 1.0 / 1.5 s | Permutation rejection 0.5 / 1.0 / 1.5 s |
|---|---|---|---|
| Metrica | v1 | 1.000 / 1.000 / 0.965 | 1.000 / 1.000 / 0.996 |
| Metrica | **v2** | 1.000 / **0.977** [0.975, 0.979] / 0.153 | 0.999 / **0.994** [0.993, 0.995] / 0.962 |
| Metrica | disc 7 m/s | 0.932 / 0.168 / 0.039 | 0.994 / 0.957 / 0.860 |
| SkillCorner | v1 | 1.000 / 1.000 / 0.955 | 1.000 / 0.999 / 0.992 |
| SkillCorner | **v2** | 1.000 / **0.960** [0.959, 0.961] / 0.205 | 1.000 / **0.988** [0.987, 0.989] / 0.926 |
| SkillCorner | disc 7 m/s | 0.938 / 0.204 / 0.061 | 0.991 / 0.920 / 0.781 |

Cost: one v2 envelope at 1.5 s takes 2.4 ms (Metrica run) / 2.2 ms (SkillCorner run), median,
against 0.7 ms for v1; vectorised throughput is about 3,400–3,700 envelopes per second
(v1 11,000–12,000).

**Decision.** Every pre-registered check passes on both providers, so v2 is the default
(`kinematics.ENVELOPE_VERSION = 2`) for the response policies' clipping and the kinematic
feasibility diagnostic, with each provider's noise margin. v1 stays selectable
(`ENVELOPE_VERSION = 1` or `default_limits(version=1)`), and every committed session-3 result
still reproduces: `response_validation`, `rule_response_check` v1/v2 and
`response_sensitivity` v1 pin `DEFAULT_LIMITS`. With v2 clipping, the learned model still
meets its forecast rule (Metrica Δ FDE −0.0088 [−0.0097, −0.0077] m at 0.5 s and −0.039
[−0.043, −0.036] at 1.0 s; SkillCorner −0.029 [−0.030, −0.028] and −0.126 [−0.129, −0.123];
better in 3/3 and 20/20 matches). v2 clipping no longer costs it anything (on SkillCorner
v1 clipping cost 0.005 m at 0.5 s).

**Which causes mattered** (coverage at 0.5 s / 1.0 s, descriptive ablations):

| Variant | Metrica | SkillCorner |
|---|---|---|
| v1 | 31.1% / 72.5% | 18.8% / 60.9% |
| v1 without the reaction delay | 88.1% / 93.4% | 80.3% / 87.2% |
| v2 structure with v1's 3.5 m/s² | 98.0% / 97.5% | 97.0% / 94.5% |
| v2 without the velocity lag | 99.5% / 99.8% | 99.6% / 99.8% |
| v2 without the noise margin | 99.7% / 99.9% | 99.9% / 99.9% |
| v2 isotropic (no force–velocity limit) | 99.8% / 99.9% | 99.98% / 99.99% |
| v2 | 99.8% / 99.9% | 99.97% / 99.96% |

1. *Reaction delay: the main cause.* Removing it alone recovers 57 / 21 points (Metrica) and
   62 / 26 points (SkillCorner). In continuous play players are already accelerating; the
   delay asserted an acceleration of zero.
2. *Acceleration bound: the second cause.* Raising 3.5 to 7 m/s² (with the force–velocity
   limit) closes the rest. Smoothed match accelerations exceed 3.5 m/s² in 1–7% of frames
   (Metrica 1.2% below 2 m/s, 5.5% at 4–6 m/s; SkillCorner 2.6% and 7.3%).
3. *Direction dependence: real, and it buys precision rather than coverage.* At speed the
   braking and turning tails exceed the propulsive tail (99th percentiles at 4–6 m/s:
   propulsive 3.8 / 3.4, braking 4.3 / 4.3, turning 4.4 / 4.5 m/s², Metrica / SkillCorner;
   the Metrica 6+ m/s band is inflated by tracking artefacts). The force–velocity limit costs
   at most 0.03 points of coverage but cuts the area by 35% at 1.0 s (60 → 39–40 m²) and lifts
   1.0 s shift rejection from 0.88 / 0.83 to 0.98 / 0.96.
4. *Velocity estimate: lag more than noise.* The causal 0.2 s velocity differs from the
   centred velocity at `t` by 0.13–0.42 m/s RMS on Metrica (0.15–0.21 SkillCorner), but from
   the centred velocity 0.1 s earlier by only 0.05–0.19 (0.03–0.04): the error is mainly the
   0.1 s lag, as expected from the estimator. Stationary-player velocity noise is small
   (σv 0.069 / 0.079 m/s pooled; robust 0.030 / 0.053), and the 10 fps estimate is only
   slightly noisier. The lag term is worth 0.3–0.5 points of coverage at 0.5 s.
5. *Measurement margin: minor.* σp is 0.017 m / 0.033 m (RMS; median 0.006 / 0.013); the
   margin is 0.18 / 0.23 m at 1.0 s and adds 0.1 points of coverage. SkillCorner positions are
   smoothed by the provider, so stationary jitter may understate their error while moving;
   the margin is a lower bound there.
6. *The horizon-end test was not wrong:* the unit conversion is correct (pinned by a test),
   and v1's misses are explained by 1–2 above.

**Precision trade-off and the scenario's `supported`.** v2 is far larger than v1 (it was an
inner bound) but rejects 96–98% of 5 m shifts at 1.0 s. At the scenario's default 1.5 s every
physical envelope becomes permissive (shift rejection 0.15 / 0.21), yet v2 still dominates the
constant-speed disc that decides `supported`: equal or higher coverage, an area 2.5× smaller
at 1.5 s (about 4× at 1.0 s), and 3–4× the shift rejection at 1.5 s (5–6× at 1.0 s). `supported` was **not** switched: the web
lab draws the disc (`reach_radius_m`), the API schema says the kinematic check "does not
change supported", and both belong to the API owner. The switch is one constant
(`scenario.SUPPORTED_REACHABILITY = "envelope"`, which also rewords the warning); with it set,
all 26 existing scenario/API tests still pass on real data (checked, then reverted). Post hoc
and not adoptable: `v2_accel_3.5` already reaches 97–98% coverage at a third of v2's area,
so a tighter, empirically sized bound would need its own pre-registration and a fresh
held-out test.

## Part 2. Why is the learned response model inert?

### Pre-registration (written and committed before the D4 evaluation was run)

`research/response_diagnostics.py`, per provider, on the `response_validation` rows
(pass-start frames, outfield defenders, detected-only on SkillCorner, 1.0 s):

* **D1, conditioning.** Relocate the attacker nearest the ball by 5 m (with the ball) in four
  directions, recompute the served model's inputs, and report the change of each standardised
  input and of the 1.0 s prediction (the model's gain in metres per metre).
* **D2, target.** R² of constant velocity for displacement; R² of the v1 inputs for the
  *innovation* (displacement beyond own constant velocity), LOMO.
* **D3, out of distribution.** The share of intervened input vectors beyond the observed rows'
  99th percentile of Mahalanobis distance; the share of real attacker innovations larger than
  the 5 m probe.
* **D4, innovation design** (`counterfactual/response_innovation.py`). Ridge (α = 1, a priori)
  of each defender's innovation on five summaries of the attackers' and the ball's innovations
  (nearest attacker by softmin over 1 m, Gaussian kernel means at 5 m and 15 m, team mean,
  ball), per frame offset to 1.0 s, LOMO. Placebo: the same model refit after permuting the
  attacker innovations among the outfield attackers of each frame. **Rule:** the model is
  *supported* on a provider iff at 0.5 s and 1.0 s the 95% CIs of FDE(CV + model) − FDE(CV)
  and of FDE(model) − FDE(placebo) are both below zero (possession-clustered bootstrap,
  1,000, stratified by match, paired rows). If it is supported on a provider, it is exported
  per provider and served as a separate, versioned policy `learned_innovation`; `learned` is
  kept unchanged and re-described as a forecaster. A runtime pilot on one SkillCorner match
  was run with its output discarded unread.
