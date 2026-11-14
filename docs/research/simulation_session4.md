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
