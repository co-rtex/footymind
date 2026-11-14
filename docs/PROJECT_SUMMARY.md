# FootyMind: project summary (for portfolios and résumés)

Every number here is traceable to [`CLAIMS_LEDGER.md`](CLAIMS_LEDGER.md) or to a test or
verification log named in [`CURRENT_STATE.md`](CURRENT_STATE.md). Do not quote numbers that are
not in those files.

## One paragraph

FootyMind is an open-source football analytics lab that reconstructs the "invisible game", the
space, pressure and passing options event data cannot see, from player-tracking data from two
providers. It covers 3 Metrica matches at 25 Hz and 20 SkillCorner A-League matches from 10 fps
broadcast tracking. It combines a leakage-audited research pipeline with an interactive web lab.
There you can replay real possessions, read metric-grounded explanations, find similar
situations from other matches, and edit a defender's position under explicit, partly validated
opponent-response assumptions.

## What is distinctive

* **Scientific honesty as a product feature.**
  * Every value carries a provenance label (observed, derived, estimated, hypothetical,
    simulated).
  * Every quoted number has a claims-ledger entry pointing to a committed results file.
  * Negative results are reported alongside positive ones.
* **Leakage control that goes beyond "don't use future frames".** The project found and fixed
  kick information hidden in broadcast event frames (a 0.2 s guard frame, calibrated
  empirically), censored unobserved inputs instead of filling them, and fixed frame-rate unit
  bugs between the 25 Hz and 10 fps providers.
* **Cross-provider replication.** Every experiment is run separately on optical and broadcast
  tracking, and never pooled.
* **Counterfactuals with stated assumptions.** Opponent-response rules are checked against
  real defender movement with a placebo design. The product labels which rules pass that check
  and which do not.

## Verified technical contributions (safe to cite)

* **Spatial context improves decision-time pass prediction on two providers:**
  * Metrica, 3 matches: Δ log loss −0.026 [−0.034, −0.018], AUC 0.648 → 0.721 (C20);
  * SkillCorner, 20 matches, 15,914 passes: −0.021 [−0.025, −0.018], better in 20/20
    held-out matches (C28);
  * unchanged when every feature is read 0.2 s before the release (C34, C35).
* **The last second of movement adds a further gain:** −0.014 (Metrica) and −0.009
  (SkillCorner, 20/20 matches) at the guard frame (C33, C35).
* **Similar-moment retrieval is validated held-out:** −0.057 and −0.051 log loss over the base
  rate (C36).
* **Per-provider calibrated option models:** held-out log loss 0.414, AUC 0.796, ECE 0.019
  (Metrica, C21) and 0.354 / 0.811 / 0.024 (SkillCorner, C24). Retrained with every feature
  read 0.2 s before the release: 0.419 / 0.790 / 0.028 and 0.354 / 0.810 / 0.021 (C71).
* **Opponent-response rules checked against real defender movement:** `marker_follows` and
  `lane_cover` beat constant velocity and a size-matched rotation placebo on both providers in a
  paired, possession-clustered test; `press_carrier` and `zonal_shift` do not, and the product
  labels them "not supported" (C70).
* **Rigorous negative results:**
  * physics pitch-control probabilities are miscalibrated (C7, C15);
  * the Defensive Shadow metric is no better than the nearest defender (C17, C29);
  * a simple off-ball run value ranks receivers near chance, far below simple baselines (C22, C25);
  * the off-ball interaction is mostly model non-additivity (C31, C32);
  * crediting a run with the defenders' reaction changes its value by almost nothing, even with
    a fitted whole-block response; the credit depends on the background convention (C45–C49).

## Engineering scope

* Python 3.13 package: provider adapters, canonical tracking model, spatial models,
  counterfactual engine, and research pipeline (scikit-learn, Polars, NumPy, FastAPI).
* TypeScript / React / Vite web lab with a Canvas pitch renderer, deep links and responsive
  layouts.
* CI on GitHub Actions: lint, type checks, unit, real-data and end-to-end tests, and a Docker
  build. Pinned, checksum-verified data downloads. A one-command reproduction script for every
  committed research artefact.
* Test counts at the end of the latest session are recorded in `CURRENT_STATE.md`.

## Résumé-ready bullets (all metrics verified; adjust wording, not numbers)

* Built a provenance-labelled football tracking analytics platform (Python/FastAPI,
  React/TypeScript) over 23 real matches from two providers. It has interactive replay,
  metric-grounded explanations, similar-situation retrieval, and counterfactual player
  repositioning with physically constrained, partly validated opponent responses.
* Designed leakage-audited, leave-one-match-out experiments showing that tracking-derived
  spatial context improves decision-time pass-outcome prediction on both optical and broadcast
  tracking (Δ log loss −0.026 and −0.021; better in 20/20 held-out broadcast matches). Also
  found and corrected kick information hidden in broadcast event frames.
* Ran pre-registered experiments with independent methodology reviews, and published negative
  results (miscalibrated physics probabilities, a defender metric no better than a nearest-
  defender baseline) alongside positive ones in a claims ledger tied to committed results.
