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
