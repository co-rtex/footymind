"""Defensive Shadow: synthetic property test and real-data API test."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from footymind.data.store import available_matches
from footymind.ghost import defensive_shadow as ds
from footymind.spatial.state import FrameState
from footymind.tracking.model import Provenance

needs_data = pytest.mark.skipif(
    "metrica-sample-1" not in available_matches(), reason="processed Metrica data not available"
)


def _synthetic_state(block: bool) -> FrameState:
    """SYNTHETIC: carrier at the origin, receiver 25 m ahead, one defender in or out of the lane."""
    xy = np.array(
        [[0.0, 0.0], [25.0, 0.0], [-45.0, 0.0], [12.5, 0.4 if block else 20.0], [50.0, 0.0]]
    )
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


def test_lane_blocking_defender_has_positive_shadow() -> None:
    v_block = ds._values(_synthetic_state(True), "physics", ds.DEFAULT_PARAMS)["home_2"]
    v_open = ds._values(_synthetic_state(False), "physics", ds.DEFAULT_PARAMS)["home_2"]
    assert v_block < v_open  # the blocking position suppresses the option


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
