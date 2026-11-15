"""Run value property tests on SYNTHETIC constructed states (not match observations)."""

import numpy as np
import pytest

from footymind.ghost.run_value import run_values_for_state
from footymind.spatial.state import FrameState
from footymind.tracking.model import Provenance


def _state(runner_now: tuple[float, float]) -> FrameState:
    # home_1 carrier at origin; home_2 the runner; home_3 GK; away_1 defender at (15, 0); away_2 GK
    xy = np.array([[0.0, 0.0], runner_now, [-45.0, 0.0], [15.0, 0.0], [50.0, 0.0]], dtype=float)
    n = xy.shape[0]
    return FrameState(
        match_id="syn",
        frame=1,
        period=1,
        time_s=0.0,
        player_ids=["home_1", "home_2", "home_3", "away_1", "away_2"],
        teams=["home", "home", "home", "away", "away"],
        jerseys=[1, 2, 3, 1, 2],
        xy=xy,
        vel=np.zeros((n, 2)),
        is_gk=np.array([False, False, True, False, True]),
        ball=np.array([0.0, 0.0]),
        attacking_team="home",
        possession_id="syn:p0",
        carrier=0,
        attack_sign=1,
        position_provenance=[Provenance.OBSERVED] * n,
        velocity_provenance=[Provenance.DERIVED] * n,
    )


def _own(now: tuple[float, float], before: tuple[float, float]) -> float:
    st = _state(now)
    earlier = st.xy.copy()
    earlier[1] = before
    rv = run_values_for_state(st, earlier, np.zeros_like(earlier))
    return next(r for r in rv if r.player_id == "home_2").own_run_value


def test_run_out_of_cover_into_space_is_positive() -> None:
    # was directly behind the defender (blocked lane), now 8 m wide of it
    assert _own(now=(25.0, 8.0), before=(25.0, 0.5)) > 0.02


def test_run_into_a_marker_is_negative() -> None:
    assert _own(now=(25.0, 0.5), before=(25.0, 8.0)) < -0.02


def test_standing_still_is_zero_and_carrier_excluded() -> None:
    st = _state((25.0, 8.0))
    rv = run_values_for_state(st, st.xy.copy(), np.zeros_like(st.xy))
    assert "home_1" not in {r.player_id for r in rv}  # the carrier is never a receiver
    assert {r.player_id for r in rv} == {"home_2", "home_3"}
    assert all(r.own_run_value == pytest.approx(0.0, abs=1e-12) for r in rv)
