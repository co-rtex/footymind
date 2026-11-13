"""API and scenario-engine behaviour on real Metrica data (skipped if absent)."""

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from footymind.counterfactual.scenario import Intervention, run_scenario
from footymind.data.store import available_matches, load_match
from footymind.spatial.state import frame_state

pytestmark = pytest.mark.skipif(
    "metrica-sample-1" not in available_matches(), reason="processed Metrica data not available"
)
MID = "metrica-sample-1"


@pytest.fixture(scope="module")
def client() -> TestClient:
    from footymind.api.app import app

    return TestClient(app)


@pytest.fixture(scope="module")
def pass_frame() -> tuple[int, str, str]:
    p = load_match(MID).events.filter(pl.col("type") == "PASS").row(100, named=True)
    return p["start_frame"], p["from_player_id"], p["to_player_id"]


def test_replay_shapes_and_nulls(client: TestClient) -> None:
    pos = client.get(f"/api/matches/{MID}/possessions", params={"min_passes": 4}).json()
    r = client.get(f"/api/possessions/{pos[0]['possession_id']}/replay").json()
    n_f, n_p = len(r["frame"]), len(r["players"])
    assert len(r["players_xy"]) == n_f * n_p * 2
    assert len(r["ball_xy"]) == n_f * 2
    assert r["provenance"] == "observed"
    # substitutes are not on the pitch early in the match -> null, never 0
    assert any(v is None for v in r["players_xy"])
    assert "Metrica" in r["source"]["name"]


def test_analysis_contract(client: TestClient, pass_frame: tuple[int, str, str]) -> None:
    f, passer, receiver = pass_frame
    a = client.get(f"/api/matches/{MID}/frames/{f}/analysis").json()
    assert a["carrier_id"] == passer
    pc = a["pitch_control"]
    assert len(pc["values"]) == pc["nx"] * pc["ny"] and pc["provenance"] == "estimated"
    assert receiver in {o["receiver_id"] for o in a["pass_options"]}
    assert all(0 <= o["p_success"] <= 1 for o in a["pass_options"])
    assert "uncalibrated" in a["model"]["status"] or "not calibrated" in a["model"]["status"]


def test_errors(client: TestClient) -> None:
    assert client.get("/api/matches/nope").status_code == 404
    assert client.get(f"/api/matches/{MID}/frames/999999999/analysis").status_code == 404
    bad = {"match_id": MID, "frame": 100, "interventions": []}
    assert client.post("/api/scenarios", json=bad).status_code == 422
    r = client.get(f"/api/matches/{MID}/replay", params={"start_frame": 1, "end_frame": 100000})
    assert r.status_code == 422


def test_scenario_does_not_mutate_observed(pass_frame: tuple[int, str, str]) -> None:
    f, _, _receiver = pass_frame
    s = frame_state(MID, f)
    before = s.xy.copy()
    d = next(i for i, t in enumerate(s.teams) if t != s.attacking_team and not np.isnan(s.xy[i, 0]))
    pid = s.player_ids[d]
    r = run_scenario(MID, f, [Intervention(pid, s.xy[d, 0] + 2.0, s.xy[d, 1])], ensemble=4)
    np.testing.assert_array_equal(frame_state(MID, f).xy, before)
    np.testing.assert_array_equal(r.observed.state.xy, before)
    assert r.hypothetical.state.position_provenance[d].value == "hypothetical"
    assert r.supported


def test_unreachable_intervention_flagged(pass_frame: tuple[int, str, str]) -> None:
    f, _, _ = pass_frame
    s = frame_state(MID, f)
    d = next(i for i, t in enumerate(s.teams) if t != s.attacking_team and not np.isnan(s.xy[i, 0]))
    far = (-s.xy[d, 0], -s.xy[d, 1])  # mirror through the centre spot
    r = run_scenario(MID, f, [Intervention(s.player_ids[d], *far)], ensemble=0)
    assert not r.supported
    assert any("unsupported" in w for w in r.warnings)


def test_marker_follows_policy(pass_frame: tuple[int, str, str]) -> None:
    f, _, receiver = pass_frame
    s = frame_state(MID, f)
    i = s.index(receiver)
    iv = Intervention(receiver, s.xy[i, 0] + 3.0, s.xy[i, 1])
    r = run_scenario(MID, f, [iv], policy="marker_follows", follow_fraction=0.5, ensemble=0)
    sim = [m for m in r.movements if m.provenance.value == "simulated"]
    assert len(sim) <= 1
    for m in sim:
        assert np.hypot(m.to_xy[0] - m.from_xy[0], m.to_xy[1] - m.from_xy[1]) == pytest.approx(1.5)


def test_scenario_endpoint_deterministic(
    client: TestClient, pass_frame: tuple[int, str, str]
) -> None:
    f, _, receiver = pass_frame
    s = frame_state(MID, f)
    i = s.index(receiver)
    body = {
        "match_id": MID,
        "frame": f,
        "ensemble": 8,
        "seed": 3,
        "interventions": [
            {"player_id": receiver, "x": float(s.xy[i, 0]) + 2, "y": float(s.xy[i, 1])}
        ],
    }
    a = client.post("/api/scenarios", json=body).json()
    b = client.post("/api/scenarios", json=body).json()
    assert a["option_deltas"] == b["option_deltas"]
    assert a["disclaimer"] and a["assumptions"]
