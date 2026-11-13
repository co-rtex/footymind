"""Mathematical properties of the PPCF model on constructed (synthetic) states."""

import itertools

import numpy as np
import pytest

from footymind.spatial.lanes import lane, point_segment_distance
from footymind.spatial.pitch_control import DEFAULT_GRID, DEFAULT_PARAMS, pitch_control

Z = np.zeros((1, 2))


def pc(target, att, dfn, ball=(0.0, 0.0), av=None, dv=None):
    att, dfn = np.atleast_2d(att).astype(float), np.atleast_2d(dfn).astype(float)
    av = np.zeros_like(att) if av is None else np.atleast_2d(av).astype(float)
    dv = np.zeros_like(dfn) if dv is None else np.atleast_2d(dv).astype(float)
    return pitch_control(
        np.atleast_2d(target).astype(float), np.array(ball, float), att, av, dfn, dv
    )


def test_symmetric_players_split_control() -> None:
    p, pa, pd = pc((10.0, 0.0), (10.0, 5.0), (10.0, -5.0))
    assert p[0] == pytest.approx(0.5, abs=0.01)
    assert pa.sum() + pd.sum() == pytest.approx(1.0, abs=1e-9)


def test_decisive_arrival() -> None:
    assert pc((10.0, 0.0), (10.0, 0.5), (40.0, 30.0))[0][0] == pytest.approx(1.0)
    assert pc((10.0, 0.0), (40.0, 30.0), (10.0, 0.5))[0][0] == pytest.approx(0.0)


def test_monotone_in_defender_distance() -> None:
    vals = [pc((10.0, 0.0), (10.0, 6.0), (10.0, -d))[0][0] for d in (2, 4, 6, 8, 12)]
    assert all(a < b for a, b in itertools.pairwise(vals))


def test_velocity_towards_target_helps() -> None:
    still = pc((10.0, 0.0), (10.0, 6.0), (10.0, -6.0))[0][0]
    running = pc((10.0, 0.0), (10.0, 6.0), (10.0, -6.0), av=(0.0, -4.0))[0][0]
    assert running > still + 0.2


def test_unobserved_players_excluded() -> None:
    base = pc((10.0, 0.0), (10.0, 5.0), (10.0, -5.0))[0][0]
    with_nan = pc((10.0, 0.0), [(10.0, 5.0), (np.nan, np.nan)], [(10.0, -5.0)])[0][0]
    assert with_nan == pytest.approx(base)


def test_grid_is_probability_field() -> None:
    rng = np.random.default_rng(1)
    att = rng.uniform([-50, -30], [50, 30], (11, 2))
    dfn = rng.uniform([-50, -30], [50, 30], (11, 2))
    p, pa, pd = pitch_control(
        DEFAULT_GRID.centres(),
        np.array([0.0, 0.0]),
        att,
        np.zeros_like(att),
        dfn,
        np.zeros_like(dfn),
    )
    assert p.shape == (DEFAULT_GRID.nx * DEFAULT_GRID.ny,)
    assert np.all((p >= 0) & (p <= 1))
    np.testing.assert_allclose(pa.sum(1) + pd.sum(1), 1.0, atol=1e-6)


def test_point_segment_distance() -> None:
    a, b = np.array([0.0, 0.0]), np.array([10.0, 0.0])
    pts = np.array([[5.0, 3.0], [-4.0, 3.0], [13.0, -4.0]])
    np.testing.assert_allclose(point_segment_distance(pts, a, b), [3.0, 5.0, 5.0])


def test_lane_blocked_vs_open() -> None:
    ball, tgt = np.array([0.0, 0.0]), np.array([40.0, 0.0])
    blocked = lane(ball, tgt, np.array([[20.0, 0.5]]), np.zeros((1, 2)))
    open_ = lane(ball, tgt, np.array([[20.0, 25.0]]), np.zeros((1, 2)))
    assert blocked.p_lane_clear < 0.2 < 0.9 < open_.p_lane_clear
    assert blocked.min_margin_s < 0 < open_.min_margin_s
    assert blocked.n_defenders_within_3m_of_lane == 1 and open_.n_defenders_within_3m_of_lane == 0
    assert blocked.ball_travel_time_s == pytest.approx(40.0 / DEFAULT_PARAMS.average_ball_speed)


def test_lane_short_pass_limitation_documented() -> None:
    """Uniform reaction time: a defender next to a short, fast pass is only a moderate threat.

    This pins the known limitation described in ``footymind.spatial.lanes`` so
    that any change to it is deliberate.
    """
    r = lane(np.array([0.0, 0.0]), np.array([12.0, 0.0]), np.array([[6.0, 0.3]]), np.zeros((1, 2)))
    assert 0.2 < r.p_lane_clear < 0.9
