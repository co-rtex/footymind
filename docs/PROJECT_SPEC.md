# FootyMind 2.0: The Invisible Game. Project specification and roadmap

## Vision
An interactive football intelligence and counterfactual-tactics laboratory built on real
tracking data. The defining question:

> Can we quantify the tactical value of decisions, positioning and off-ball movement that
> conventional match statistics overlook, and do so in a way that is scientifically
> defensible?

## Principles
1. **Observed playback is real.** Synthetic data is used only for tests and is labelled as such.
2. **Provenance everywhere.** Observed, derived, estimated, hypothetical and simulated
   values are distinct in code, the API and the UI.
3. **Counterfactuals are model-based, not causal claims.** Every scenario carries
   assumptions, feasibility checks and a parameter-sensitivity ensemble.
4. **Complexity is earned.** Physics and linear models come first. Expressive models only
   follow controlled evidence (ADR-0003).
5. **Negative results are reported.** CLAIMS_LEDGER.md is the only source of quotable numbers.

## Flagship capabilities
* **Ghost Value**: value of actions and positions that do not touch the ball (space
  creation, lane blocking, decoy runs). *Status*: building blocks (pitch control, lane
  geometry, pressure, interventions) are implemented. A validated off-ball valuation is
  not yet implemented.
* **Possession Multiverse**: open a real possession, pause, inspect options, move a player
  within physically reachable limits, compare observed with hypothetical outcomes,
  including uncertainty. *Status*: backend complete (v1). Frontend in progress.

Supporting features (Football DNA, AI Film Room, Defensive Shadow, Anti-Highlight Reel,
Tactical Scientist) are deferred until the flagship demo is solid.

## Roadmap

### Done (Sprint A–C core, 2026-11-13)
- [x] Legacy audit and archive; repository hygiene
- [x] Metrica ingestion (games 1–2), canonical model, quality gates, possessions
- [x] Replay / analysis / scenario API with provenance
- [x] Pitch control, lanes, pressure; constrained interventions with sensitivity ensemble
- [x] Pass-outcome research v1: A/B/C families, ablations, sensitivity analyses

### Next (priority order)
1. **Frontend flagship demo** (replay → pause → analyse → drag → compare → reset), with
   e2e tests and screenshots. *(in progress)*
2. **CI** running lint, types, unit tests, data tests (downloads Metrica) and the web build.
3. **Metrica Game 3 (EPTS) adapter**, giving a third LOMO fold. Re-run pass_outcome with 3 matches.
4. **Calibrate the physics model on data**: fit PPCF parameters (reaction time, σ, λ, ball
   speed) by maximum likelihood on completed vs. failed passes in training folds only, then
   compare with the literature parameters. The scenario engine can then switch to calibrated
   parameters with honest held-out evidence.
5. **Defensive Shadow v1** (first Ghost Value metric): for each defender at a frame, the
   reduction in the attacking team's option value relative to the defender's
   reachable-set alternatives (a displacement-sensitivity analysis, not removal, because
   removing a player is unphysical). Validate against the observed pass outcomes it implies.
6. **Lead-pass targets**: evaluate options at the receiver's reachable future positions
   rather than only their current position.
7. SkillCorner integration with explicit visibility modelling. Cross-provider evaluation
   reported separately.
8. Retrieval of tactically similar situations (Football DNA groundwork).

### Not planned without new evidence
Large generative world models, diffusion trajectory models, RL agents and GNNs (see ADR-0003).
