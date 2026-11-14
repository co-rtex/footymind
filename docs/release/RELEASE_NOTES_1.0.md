# FootyMind 1.0 release notes

FootyMind is a football tracking-analytics lab. It reconstructs the space, pressure and passing
options the event log cannot see, from real player tracking by two providers (3 Metrica
matches, optical, 25 Hz; 20 SkillCorner A-League matches, broadcast, 10 fps). It pairs a
leakage-audited research pipeline with an interactive web lab: replay, evidence-grounded
explanations, similar moments, and "what if" repositioning under stated, partly validated
assumptions about how defenders respond.

Every number below is traced in [`docs/CLAIMS_LEDGER.md`](../CLAIMS_LEDGER.md) to a committed
result or a test.

## What is in 1.0

**Data platform.** Checksum-pinned downloads; adapters for Metrica CSV and EPTS and SkillCorner
broadcast tracking; a canonical model with provenance (observed, derived, estimated,
hypothetical, simulated) and camera-detection masks; versioned bundles (pipeline v3); data
quality gates and a measured data-quality audit (C30).

**Analysis engine.** Pitch control and lane geometry (labelled uncalibrated); per-provider
calibrated pass-option models (C21, C24, C71); causal temporal context with a 0.2 s guard frame
(ADR-0007, C38); evidence-grounded "Why this moment" statements; similar-moment retrieval with
a validated k = 50 estimate (C36).

**Possession Multiverse.** Relocate up to four players at a real frame under a reachability check
and a kinematic envelope (C55–C58); seven opponent-response policies, each served with a
status sentence, a machine-readable verdict and, where fitted, its fitted range; a
parameter-sensitivity ensemble; observed vs hypothetical vs difference views.

**Web lab.** A guided tour over three featured real situations (two Metrica, one SkillCorner),
Canvas replay, keyboard and drag editing, provenance styling (hollow = estimated), policy verdict
badges, similar moments, the experimental Defensive Shadow layer, responsive layouts and
deep links. Featured situations whose data is not loaded are hidden with a note.

**Operations.** One Docker image (API plus lab, no data baked in); an entrypoint that downloads
and ingests on first start and repairs an existing volume (outdated bundles, interrupted first
starts, stale indexes), verified in the built image in CI; cache freshness after re-ingest;
bounded, validated API inputs; per-provider model isolation (C69).

## Research results

* **Spatial context helps before the pass is played.** Decision-time Δ log loss −0.026
  [−0.034, −0.018] on Metrica (C20) and −0.021 [−0.025, −0.018] on SkillCorner with every
  feature read 0.2 s before the kick (15,893 passes, 20/20 held-out matches, C35).
* **The last second of movement adds more:** −0.014 (Metrica) and −0.009 (SkillCorner, 20/20)
  at the guard frame (C33, C35).
* **Similar moments are informative:** the k = 50 neighbour rate beats the base rate held-out
  (−0.057 / −0.051 log loss, C36).
* **Response rules checked against real reactions:** `marker_follows` and `lane_cover` pass a
  paired observational check against constant velocity and size-matched placebos on both
  providers; `press_carrier` and `zonal_shift` do not (C70).
* **Negative results, kept:** literature pitch-control probabilities are miscalibrated (C7,
  C15); the Defensive Shadow is no better than the nearest defender (C17, C29); a simple
  run value ranks receivers near chance (C22, C25); crediting a run with defenders' reactions
  changes its modelled value by almost nothing, and the credit depends on the background
  convention (C45–C49).

Full write-up: [`docs/RESEARCH_REPORT.md`](../RESEARCH_REPORT.md); models:
[`docs/MODEL_CARDS.md`](../MODEL_CARDS.md).

## Changes in the final release session

* `scripts/container_recovery.py`: the entrypoint's recovery on existing volumes (healthy
  restart, outdated bundle, interrupted first start, unknown source and unreadable file),
  run in the built image in CI.
* The CLI names its sources explicitly (`metrica`, `skillcorner`, `all` or match ids); a bare
  `footymind download` no longer fetches about 2 GB. The README gives four setup paths with
  measured sizes; a test checks every documented download/ingest command.
* The entrypoint expands source groups on existing volumes too, which makes the licence-clear
  SkillCorner-only demo configuration work (verified).
* `docs/release/DATA_LICENSES.md`, `DEPLOYMENT.md`, `docs/ROADMAP.md`, this file and the
  release checklist.
* Response models labelled for another provider are refused (C69); claims-ledger and
  documentation corrections from an independent research-integrity audit; product-QA fixes
  (see [`docs/product/release-1.0/QA_REPORT.md`](../product/release-1.0/QA_REPORT.md)).
* `scripts/` is linted and type-checked in CI.

## Known limitations

* Three Metrica matches: Metrica intervals reflect within-match sampling only. One
  SkillCorner league-season.
* Broadcast tracking extrapolates off-camera players; those positions are labelled estimated
  and never presented as observed.
* What-if results are model-based comparisons, never causal claims. Response policies are
  assumptions with an observational check, not causal models; `learned` is nearly inert under
  edits; `learned_innovation` extrapolates beyond about 2.3 / 2.5 m of unexpected movement.
* The option model is fitted on attempted passes with inferred targets; options nobody chose
  are extrapolations.
* The API has no authentication or rate limiting; put a reverse proxy in front of a public
  deployment.
* A data file damaged under a valid manifest is not repaired automatically; it answers 503
  with the command that fixes it.

## Data and licences

Raw data is never committed and the image contains none. SkillCorner open data is MIT-licensed
(credit requested); Metrica's sample data has no licence file and asks for responsible use and
acknowledgement. A SkillCorner-only public-demo configuration avoids the open question. See
[`DATA_LICENSES.md`](DATA_LICENSES.md).

## Upgrading

Start the 1.0 image on an existing volume: bundles from an earlier pipeline version are
re-ingested and stale indexes rebuilt automatically. Locally, run
`uv run footymind ingest <sources>` and `uv run footymind build-similarity-index`.
