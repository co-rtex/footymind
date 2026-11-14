# ADR-0006: Opponent response policies with kinematic limits

- Status: accepted (2026-11-14)

## Context
The Possession Multiverse relocated players on a single observed frame and offered one toy
response rule (`marker_follows`). Off-ball valuation claims need defenders that react, but a
simulated reaction must not be mistaken for what a defender would have done, and a
physically reachable move is not a tactically plausible one.

## Decision
1. **Kinematic envelope** (`counterfactual/kinematics.py`). A bounded double integrator: the
   player keeps the observed causal velocity for a reaction delay (0.25 s), then accelerates
   at most 3.5 m/s² in any direction with speed at most 7.5 m/s. The reachable set is
   approximated from the inside by constant-heading full-thrust paths (48 headings). The
   constants are a priori and named; an unknown velocity falls back to a start from rest and
   says so.
2. **Policies** (`counterfactual/response.py`): `none`, `marker_follows` (legacy,
   bit-for-bit), `press_carrier`, `zonal_shift`, `lane_cover` (rules) and `learned`
   (ridge model of 0.04–1.0 s defender displacement, one plain-JSON coefficient file per
   provider). Only outfield defenders move; attackers, goalkeepers and user-moved players
   never do. Every move is `SIMULATED` and carries a reason.
3. **Differential responses.** A policy maps a state to target positions after the horizon,
   each target is projected onto the defender's envelope, and the applied move is
   `clip(target | hypothetical) − clip(target | observed)`. The analysed instant stays the
   observed frame, nothing moves without an intervention, and the response is stable,
   because projection onto a convex set is non-expansive.
4. **Validation** (`research/response_validation.py`), per provider and never pooled: forecast
   where outfield defenders actually went 0.5 s and 1.0 s after pass-start frames. The
   comparators are stationary, constant velocity, clipped constant velocity, learned and
   learned + clip. The protocol is LOMO with a fixed ridge alpha and a possession-clustered
   bootstrap. On broadcast tracking only detected positions count, and exclusions are
   counted.
5. **Pre-specified rule:** `validated = true` for `learned` on a provider iff the 95% CI of
   FDE(learned + clip) − FDE(constant velocity) is below zero at both horizons. The rule
   policies are always `validated = false` (experimental): nothing observable tests them.

## Consequences
* Results (docs/RESEARCH.md, C40–C44): `learned` meets the rule on both providers, but
  it is close to damped constant velocity and responds to interventions by centimetres.
* The validation is observational. It tests forecasts of real play, never responses to
  hypothetical interventions, for which no ground truth exists. The API states this next to
  `validated`.
* The envelope is a deliberately tight inner bound. Its coverage of real positions is
  reported (C41, C43), and it is a diagnostic for user moves, never a gate on `supported`.
* The `learned` policy is linear and mirror-equivariant by construction. It cannot
  represent coordinated team behaviour; a richer model (e.g. graph-based ghosting) would need
  the same differential and kinematic wrapper and the same validation.
