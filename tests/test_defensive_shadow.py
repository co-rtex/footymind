"""Defensive Shadow: property tests on SYNTHETIC constructed states + a real-data API test.

Synthetic states are hand-built geometry, not match observations.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from footymind.data.store import available_matches
from footymind.ghost.defensive_shadow import _values, shadow_for_state
from footymind.spatial.pitch_control import DEFAULT_PARAMS
from footymind.spatial.state import FrameState
from footymind.tracking.model import Provenance

needs_data = pytest.mark.skipif(
    "metrica-sample-1" not in available_matches(), reason="processed Metrica data not available"
)


def _state(
    defenders: list[tuple[float, float]], gk_def: tuple[float, float] = (50.0, 0.0)
) -> FrameState:
    """Carrier at the origin; receivers 25 m ahead (home_2) and 20 m wide (home_3)."""
    att = [(0.0, 0.0), (25.0, 0.0), (0.0, 20.0), (-45.0, 0.0)]
    xy = np.array(att + list(defenders) + [gk_def], dtype=float)
    n = xy.shape[0]
    n_def = len(defenders) + 1
    return FrameState(
        match_id="syn",
        frame=1,
        period=1,
        time_s=0.0,
        player_ids=[f"home_{i + 1}" for i in range(4)] + [f"away_{i + 1}" for i in range(n_def)],
        teams=["home"] * 4 + ["away"] * n_def,
        jerseys=list(range(1, n + 1)),
        xy=xy,
        vel=np.zeros((n, 2)),
        is_gk=np.array([False, False, False, True] + [False] * (n_def - 1) + [True]),
        ball=np.array([0.0, 0.0]),
        attacking_team="home",
        possession_id="syn:p0",
        carrier=0,
        attack_sign=1,
        position_provenance=[Provenance.OBSERVED] * n,
        velocity_provenance=[Provenance.DERIVED] * n,
    )


def _shadow(st: FrameState) -> dict[str, object]:
    r = shadow_for_state(st, value_model="physics")
    return {d.player_id: d for d in r.defenders}


def test_lane_blocker_has_positive_shadow_on_that_option() -> None:
    sh = _shadow(_state([(12.5, 0.5)]))["away_1"]
    assert sh.shadow_sum > 0.05  # type: ignore[attr-defined]
    assert sh.most_suppressed_receiver == "home_2"  # type: ignore[attr-defined]


def test_isolated_defender_has_near_zero_shadow() -> None:
    sh = _shadow(_state([(-30.0, -30.0)]))["away_1"]
    assert abs(sh.shadow_sum) < 0.01  # type: ignore[attr-defined]


def test_blocker_shadow_exceeds_bystander_shadow() -> None:
    sh = _shadow(_state([(12.5, 0.5), (-30.0, -30.0)]))
    assert sh["away_1"].shadow_sum > sh["away_2"].shadow_sum + 0.05  # type: ignore[attr-defined]


def test_goalkeeper_excluded_and_alternatives_inside_pitch() -> None:
    r = shadow_for_state(_state([(52.0, 33.5)]), value_model="physics")
    assert "away_2" not in {d.player_id for d in r.defenders}  # the GK
    for d in r.defenders:
        for ax, ay in d.alternatives:
            assert abs(ax) <= 52.5 and abs(ay) <= 34.0


def test_requires_possession_and_ball() -> None:
    st = _state([(12.5, 0.5)])
    st.ball = np.array([np.nan, np.nan])
    with pytest.raises(ValueError, match="ball"):
        shadow_for_state(st)


def test_blocking_position_lowers_option_value() -> None:
    v_block = _values(_state([(12.5, 0.4)]), "physics", DEFAULT_PARAMS)["home_2"]
    v_open = _values(_state([(12.5, 20.0)]), "physics", DEFAULT_PARAMS)["home_2"]
    assert v_block < v_open


@needs_data
def test_shadow_endpoint() -> None:
    from footymind.api.app import app

    c = TestClient(app)
    r = c.get("/api/matches/metrica-sample-1/frames/20666/defensive-shadow")
    assert r.status_code == 200
    j = r.json()
    assert j["defenders"] and all(d["provenance"] == "estimated" for d in j["defenders"])
    assert len(j["defenders"][0]["alternatives"]) == 8
    assert "experimental" in j["status"]
