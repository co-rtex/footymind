"""Temporal context features: unit tests on SYNTHETIC arrays plus causality checks on real data."""

import copy

import numpy as np
import polars as pl
import pytest

from footymind.data.store import available_matches, load_match
from footymind.research.pass_dataset import DatasetConfig, _prev_event_features, label_events
from footymind.spatial.state import frame_state
from footymind.spatial.temporal import (
    MAX_CONTROL_S,
    control_duration_s,
    lookback_suffix,
    movement_features,
    temporal_context,
    track_feature_names,
)

FPS = 25.0


def test_lookback_suffix() -> None:
    assert lookback_suffix(1.0) == "_l10" and lookback_suffix(0.5) == "_l05"
    assert track_feature_names(2.0)[0].endswith("_l20")


def test_control_duration_counts_back_and_stops() -> None:
    t = 40
    passer = np.zeros((t, 2))
    ball = np.full((t, 2), 10.0)
    ball[-12:] = [0.5, 0.0]  # within reach for the last 12 frames
    assert control_duration_s(passer, ball, FPS) == pytest.approx(12 / FPS)
    ball[-1] = [5.0, 0.0]  # out of reach at the decision frame
    assert control_duration_s(passer, ball, FPS) == 0.0
    ball[-1] = [0.5, 0.0]
    ball[-5] = np.nan  # a run cut short by an unobserved ball is censored: unknown
    assert np.isnan(control_duration_s(passer, ball, FPS))
    ball[-5] = [9.0, 0.0]  # cut short by the ball being out of reach: a real value
    assert control_duration_s(passer, ball, FPS) == pytest.approx(4 / FPS)
    # unobserved passer or ball at the decision frame: unknown, never a fabricated 0
    gone = passer.copy()
    gone[-1] = np.nan
    assert np.isnan(control_duration_s(gone, ball, FPS))
    ball_gone = ball.copy()
    ball_gone[-1] = np.nan
    assert np.isnan(control_duration_s(passer, ball_gone, FPS))
    long_p, long_b = np.zeros((400, 2)), np.zeros((400, 2))
    assert control_duration_s(long_p, long_b, FPS) == MAX_CONTROL_S


def _scene(sign: int = 1) -> dict:
    """Synthetic: passer at x=0; one defender closes from 6 m to 2 m; attackers move forward."""
    xy_then = np.array([[0.0, 0.0], [10.0, 5.0], [6.0, 0.0], [20.0, 0.0], [20.0, 10.0]])
    xy_now = np.array([[1.0, 0.0], [12.0, 5.0], [3.0, 0.0], [19.0, 0.0], [19.0, 10.0]])
    path = np.linspace([0.0, 0.0], [1.0, 0.0], 26)
    if sign < 0:  # the same scene with the attack running toward -x
        xy_then, xy_now, path = xy_then * [-1, 1], xy_now * [-1, 1], path * [-1, 1]
    att = np.array([True, True, False, False, False])
    outfield_def = ~att
    return dict(
        xy_then=xy_then,
        xy_now=xy_now,
        ball_path=path,
        att=att,
        outfield_def=outfield_def,
        passer=0,
        sign=sign,
        lookback_s=1.0,
    )


def test_movement_features_values() -> None:
    f = movement_features(**_scene())
    assert f["press_nearest_def_change_m_l10"] == pytest.approx(2.0 - 6.0)
    assert f["press_def_within_5m_change_l10"] == 1
    assert f["ball_travel_m_l10"] == pytest.approx(1.0)
    assert f["passer_fwd_m_l10"] == pytest.approx(1.0)
    assert f["att_centroid_fwd_m_l10"] == pytest.approx(1.5)
    assert f["def_centroid_fwd_m_l10"] == pytest.approx(-5.0 / 3)


def test_movement_features_mirror_invariant() -> None:
    a, b = movement_features(**_scene(1)), movement_features(**_scene(-1))
    for k in a:
        assert a[k] == pytest.approx(b[k], nan_ok=True), k


def test_movement_features_nan_history() -> None:
    sc = _scene()
    sc["xy_then"][0] = np.nan  # passer unobserved a second ago
    f = movement_features(**sc)
    assert np.isnan(f["passer_fwd_m_l10"]) and np.isnan(f["press_nearest_def_change_m_l10"])
    assert np.isfinite(f["def_centroid_fwd_m_l10"])


MET = [m for m in available_matches() if m.startswith("metrica-")]


def _some_passes(mid: str, n: int = 6) -> list[dict]:
    b = load_match(mid)
    ev = label_events(_prev_event_features(b.events), DatasetConfig())
    return list(ev.sample(n, seed=3).iter_rows(named=True))


@pytest.mark.skipif(not MET, reason="Metrica data not ingested")
def test_temporal_context_is_causal() -> None:
    """Corrupting every frame after the decision frame must not change any feature."""
    mid = MET[0]
    td = load_match(mid).tracking
    rng = np.random.default_rng(0)
    for r in _some_passes(mid):
        i0 = td.index_of_frame(int(r["start_frame"]))
        s = frame_state(mid, int(r["start_frame"]))
        pi = td.player_index(r["from_player_id"])
        before = temporal_context(td, i0, r["team"], pi, s.is_gk, (0.5, 1.0, 2.0))
        bad = copy.copy(td)
        bad.xy = td.xy.copy()
        bad.ball_xy = td.ball_xy.copy()
        bad.xy[i0 + 1 :] = rng.normal(0, 30, bad.xy[i0 + 1 :].shape)
        bad.ball_xy[i0 + 1 :] = rng.normal(0, 30, bad.ball_xy[i0 + 1 :].shape)
        after = temporal_context(bad, i0, r["team"], pi, s.is_gk, (0.5, 1.0, 2.0))
        assert before.keys() == after.keys()
        for k in before:
            assert before[k] == pytest.approx(after[k], nan_ok=True), k


@pytest.mark.skipif(not MET, reason="Metrica data not ingested")
def test_dataset_rows_match_direct_computation() -> None:
    """Train/serve consistency: dataset columns equal the direct temporal_context call."""
    from footymind.research.pass_dataset import build

    mid = MET[0]
    df = build([mid]).head(25)
    td = load_match(mid).tracking
    for r in df.iter_rows(named=True):
        i0 = td.index_of_frame(int(r["frame"]))
        s = frame_state(mid, int(r["frame"]))
        direct = temporal_context(
            td, i0, r["team"], td.player_index(r["passer_id"]), s.is_gk, (1.0,)
        )
        for k, v in direct.items():
            assert r[k] == pytest.approx(v, nan_ok=True), k
    # time on ball is non-negative and bounded; quick_release agrees with it
    tob = df["passer_control_s"].drop_nans()
    assert (tob >= 0).all() and (tob <= MAX_CONTROL_S).all()
    ok = df.filter(pl.col("passer_control_s").is_not_nan())
    assert ((ok["passer_control_s"] < 1.0).cast(int) == ok["quick_release"].cast(int)).all()


SC = [m for m in available_matches() if m.startswith("skillcorner-")]


@pytest.mark.skipif(not SC, reason="SkillCorner data not ingested")
def test_opt_in_censoring_of_extrapolated_ball() -> None:
    """Default off (committed results unchanged); on, extrapolated balls censor time-on-ball."""
    mid = SC[0]
    td = load_match(mid).tracking
    if td.ball_detected is None:
        pytest.skip("bundle ingested before the ball_detected mask existed")
    n_diff = n_nan_more = 0
    for r in _some_passes(mid, 40):
        i0 = td.index_of_frame(int(r["start_frame"]))
        s = frame_state(mid, int(r["start_frame"]))
        p = td.player_index(r["from_player_id"])
        off = temporal_context(td, i0, r["team"], p, s.is_gk)["passer_control_s"]
        on = temporal_context(td, i0, r["team"], p, s.is_gk, censor_undetected_ball=True)[
            "passer_control_s"
        ]
        if not (off == on or (np.isnan(off) and np.isnan(on))):
            n_diff += 1
            n_nan_more += int(np.isnan(on) and not np.isnan(off))
    assert n_diff == n_nan_more  # censoring only ever turns a value into "unknown"


def _synthetic_td(ball_detected: np.ndarray | None) -> tuple:
    """SYNTHETIC tracking (constructed, not provider data): one home passer at the origin and
    one away defender; the ball sits at the passer's feet for the last 30 of 60 frames."""
    from footymind.tracking.model import PitchSpec, PlayerInfo, TrackingData

    t = 60
    xy = np.zeros((t, 2, 2))
    xy[:, 1] = [10.0, 0.0]
    ball = np.full((t, 2), 8.0)
    ball[-30:] = [0.5, 0.0]
    td = TrackingData(
        match_id="synthetic",
        pitch=PitchSpec(),
        fps=10.0,
        frame=np.arange(t),
        period=np.ones(t, dtype=np.int8),
        time_s=np.arange(t) / 10.0,
        players=[PlayerInfo("home_1", "home", 1, "1"), PlayerInfo("away_1", "away", 1, "1")],
        xy=xy,
        ball_xy=ball,
        home_attacks_positive_x={1: True},
        ball_detected=ball_detected,
    )
    return td, np.array([False, False])


def _tob(td: object, gk: np.ndarray, censor: bool, i0: int = 59) -> float:
    return float(
        temporal_context(td, i0, "home", 0, gk, censor_undetected_ball=censor)[  # type: ignore[arg-type]
            "passer_control_s"
        ]
    )


def test_censor_flag_synthetic_detected_ball_unchanged() -> None:
    """SYNTHETIC: with every ball frame detected, the flag changes nothing."""
    td, gk = _synthetic_td(np.ones(60, dtype=bool))
    assert _tob(td, gk, False) == pytest.approx(3.0)
    assert _tob(td, gk, True) == pytest.approx(3.0)
    td_none, gk = _synthetic_td(None)  # no mask (Metrica, old bundles): flag is a no-op
    assert _tob(td_none, gk, True) == pytest.approx(3.0)


def test_censor_flag_synthetic_extrapolated_ball() -> None:
    """SYNTHETIC: an extrapolated ball inside the run censors it (NaN) only when the flag is on;
    an extrapolated ball at the decision frame is unknown; extrapolation before a detected
    out-of-reach frame does not matter, because the run already ended at a detected frame."""
    det = np.ones(60, dtype=bool)
    det[45] = False  # inside the 30-frame run
    td, gk = _synthetic_td(det)
    assert _tob(td, gk, False) == pytest.approx(3.0)  # default: extrapolated counts as observed
    assert np.isnan(_tob(td, gk, True))
    det = np.ones(60, dtype=bool)
    det[59] = False  # the decision frame itself
    td, gk = _synthetic_td(det)
    assert _tob(td, gk, False) == pytest.approx(3.0)
    assert np.isnan(_tob(td, gk, True))
    assert np.isnan(
        temporal_context(td, 59, "home", 0, gk, censor_undetected_ball=True)["quick_release"]
    )
    det = np.ones(60, dtype=bool)
    det[:20] = False  # only frames before the detected out-of-reach frame 29 are extrapolated
    td, gk = _synthetic_td(det)
    assert _tob(td, gk, True) == pytest.approx(3.0)
    det = np.ones(60, dtype=bool)
    det[29] = False  # the frame that ends the run is extrapolated: the end is not observed
    td, gk = _synthetic_td(det)
    assert _tob(td, gk, False) == pytest.approx(3.0)
    assert np.isnan(_tob(td, gk, True))


def test_censor_flag_synthetic_movement_features_unaffected() -> None:
    """SYNTHETIC: the flag only touches time-on-ball; movement history is unchanged."""
    det = np.ones(60, dtype=bool)
    det[50:] = False
    td, gk = _synthetic_td(det)
    off = temporal_context(td, 59, "home", 0, gk)
    on = temporal_context(td, 59, "home", 0, gk, censor_undetected_ball=True)
    for k in track_feature_names():
        assert off[k] == pytest.approx(on[k], nan_ok=True), k


def test_event_carrier_tolerance_is_a_time() -> None:
    """Regression (session-4 audit): 3 frames meant 0.12 s at 25 Hz but 0.3 s at 10 fps."""
    from footymind.spatial.state import event_carrier_tolerance_frames

    assert event_carrier_tolerance_frames(10.0) == 3  # SkillCorner unchanged
    assert event_carrier_tolerance_frames(25.0) == 8  # Metrica now the same 0.3 s window
