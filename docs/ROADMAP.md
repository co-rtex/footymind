# Post-1.0 roadmap

FootyMind 1.0 is feature-frozen. These are research and product ideas for after 1.0. None of
them is unfinished 1.0 work, and none is promised. Each would need its own pre-registration and
claims-ledger entries before any result is quoted.

## Research

* **Multi-attacker responses.** The off-ball study (C45–C49) moves one attacker at a time, but
  real runs happen together. A joint counterfactual with the `learned_innovation` response is
  the natural next study.
* **A possession-value target.** The best-progressive-option measure barely reacts to small
  defender moves, so off-ball credit would be better measured in possession value. Response
  models identified from natural experiments would also help.
* **Responses that matter at edit scale.** `learned_innovation` is a co-movement forecaster
  fitted on real unexpected movement (99% below 2.3 / 2.5 m in 1 s). Applying it to larger user
  edits extrapolates.
* **More optical tracking.** Three Metrica matches limit every Metrica interval to
  within-match sampling.
* **More leagues and providers**, to test generalisation beyond one season.
* **A tighter kinematic envelope**, pre-registered and evaluated on fresh frames (not pass
  starts), with a shift-size curve against a size-matched disc.

## Product and operations

* An automatic repair of a bundle whose data file is damaged under a valid manifest. Today it
  answers 503 with the command that fixes it (`footymind ingest <id>`).
* Authentication and edge rate limiting for a service with real traffic (the server has an
  in-process per-client limit and a heavy-request cap; see `docs/release/DEPLOYMENT.md`).
