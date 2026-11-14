# Visibility-aware time-on-ball on SkillCorner (session 4)

Config: `experiments/configs/temporal_guard_skillcorner_v2_ballcensor.json` (pre-registered in
its description, committed at 794aab8 before the run). It is the primary SkillCorner RQ2 run
(`temporal_guard_skillcorner_v1_02s`, C35) with one change, `censor_undetected_ball = true`:
time-on-ball (`passer_control_s`, `quick_release`) is NaN when the backward count reaches a
frame where SkillCorner *extrapolated* the ball, instead of counting the extrapolated ball as
observed. Run from clean revision 66c16db.

## Behaviour of the flag (tests)
`tests/test_temporal.py` (SYNTHETIC constructed tracking, plus one real-data check):
detected balls and bundles without the mask are unchanged; an extrapolated ball inside the run,
at the frame that ends it, or at the decision frame censors the value (NaN, and
`quick_release` NaN); extrapolated frames before a detected out-of-reach frame do not matter;
movement-history features are unaffected; on real data censoring only ever turns a value into
NaN.

## Missingness (`experiments/results/ball_censor_missingness_v1_skillcorner`)
Same 15,893 rows with the flag off and on (checked).

| | flag off | flag on |
|---|---|---|
| `passer_control_s` unknown | 121 (0.76%) | 755 (4.75%); per match 3.4–6.3% |
| `quick_release` share (known rows) | 0.451 | 0.445 |

634 rows became NaN (0 changed any other way). Their uncensored mean was 1.35 s. The ball is
extrapolated at the guard frame itself in 2.2% of rows.

## Results (Δ log loss, possession-clustered 95% CI; match-level two-stage CI)

| comparison | v1 guard (C35) | v2 ball-censored | v2 match-level, matches better |
|---|---|---|---|
| `Adt → Adt_tob` (H-T1) | −0.0022 [−0.0034, −0.0012] | −0.0025 [−0.0037, −0.0014] | [−0.0041, −0.0009], 17/20 (v1: 15/20) |
| `Bdt_tob → Bdt_tt` (H-T2, LR) | −0.0091 [−0.0113, −0.0071] | −0.0091 [−0.0113, −0.0072] | [−0.0119, −0.0064], 20/20 |
| `Bdt_tob → Bdt_tt` (GBM) | −0.0089 [−0.0110, −0.0069] | −0.0087 [−0.0108, −0.0067] | |
| `Adt_tob → Bdt_tob` (H-T3) | −0.0196 [−0.0226, −0.0164] | −0.0193 [−0.0223, −0.0161] | [−0.0236, −0.0146], 20/20 |
| `Adt → Bdt` (C28) | −0.0214 [−0.0245, −0.0180] | −0.0214 (identical) | [−0.0255, −0.0166], 20/20 |

## Reading
* As pre-registered, C28 reproduces exactly (no time-on-ball in `Adt` or `Bdt`), and H-T2 and
  H-T3 move by at most 0.0003.
* H-T1 is slightly *larger* with censoring (−0.0025 vs −0.0022; 17/20 matches better instead
  of 15/20). The difference is well inside the bootstrap noise of either run, so the
  time-on-ball result does not depend on treating extrapolated balls as observed. The two runs
  share rows, so a paired v1 − v2 interval would be tighter; it was not pre-registered and is
  not computed here.
* Limitation: LR models median-impute the extra 4% NaN without a missingness indicator; the
  GBM handles NaN natively.
