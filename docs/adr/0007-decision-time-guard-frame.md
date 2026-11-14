# ADR-0007: Decision-time features read a guard frame before the release

- Status: accepted (2026-11-14)
- Supersedes nothing; it tightens ADR-0005's "decision frame".

## Context
ADR-0005 defines decision-time features at the event's start frame (the release frame). An
independent review of the temporal features (RQ2, session 3) found that on SkillCorner
(10 fps broadcast tracking) the ball is already moving away from the passer at more than
3 m/s at that frame in 31.9% of passes, against 1.3% on Metrica (25 Hz). The audit
re-measures this reproducibly in `experiments/results/data_quality_v1`.
Release-frame positions, pressure and recent-movement features can therefore carry the
start of the kick. Recomputing the temporal features 1–2 frames earlier removed about half
of SkillCorner's time-on-ball gain and about a fifth of its movement-history gain, while
Metrica's gains stayed flat. A test that corrupts every *later* frame cannot catch this,
because the contaminating frame is the decision frame itself.

## Decision
* `DatasetConfig` has two guards: `static_offset_s` (positions, pressure, structure) and
  `temporal_offset_s` (time-on-ball and movement history). Both default to 0, so committed
  results reproduce exactly.
* For SkillCorner, any decision-time claim is made from a run with both guards at **0.2 s**
  (2 frames). Release-frame results are reported only next to the guarded ones.
* Metrica guarded runs (0.2 s, 0.4 s) are reported as robustness checks.
* Features that are served for decisions which have not happened yet (similar-moment
  retrieval) always use the guarded situation (`research/similarity.py`, `GUARD_S`).
* Time-on-ball is censored (NaN) when an unobserved ball, rather than the ball leaving the
  passer's reach, ends the count.

## Consequences
* Guarded rows can differ slightly from release-frame rows. Rows whose guarded frame has no
  defender positions are skipped.
* "Decision time" now means "0.2 s before the event's start frame" for broadcast data. For
  a real decision made earlier than that, all features remain causal.
* Future features must be tested at the guarded frame, not only for "no later frame read".
