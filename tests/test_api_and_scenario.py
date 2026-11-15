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
    # the reach radius the lab draws is stated with its horizon and speed bound
    h, v = a["reach_horizon_s"], a["reach_speed_mps"]
    assert h > 0 and v > 0
    radii = [p["reach_radius_m"] for p in a["players"] if p["reach_radius_m"] is not None]
    assert radii and all(r == pytest.approx(h * v) for r in radii)


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


# --- regressions reported by the frontend review (2026-11-13) ---


def _absent_player_frame() -> tuple[int, str]:
    """A frame where a registered player is not on the pitch (a substitute)."""
    td = load_match(MID).tracking
    i = td.player_index("home_1")  # observed in ~32% of frames in game 1
    row = int(np.flatnonzero(np.isnan(td.xy[:, i, 0]) & (td.period == 2))[100])
    return int(td.frame[row]), "home_1"


def test_absent_player_velocity_provenance_is_derived(client: TestClient) -> None:
    f, pid = _absent_player_frame()
    a = client.get(f"/api/matches/{MID}/frames/{f}/analysis").json()
    p = next(p for p in a["players"] if p["player_id"] == pid)
    assert p["x"] is None and p["vx"] is None
    assert p["velocity_provenance"] == "derived"  # null value = could not be derived


def test_scenario_rejects_absent_player(client: TestClient) -> None:
    f, pid = _absent_player_frame()
    body = {"match_id": MID, "frame": f, "interventions": [{"player_id": pid, "x": 0.0, "y": 0.0}]}
    r = client.post("/api/scenarios", json=body)
    assert r.status_code == 422 and "not on the pitch" in r.json()["detail"]


def test_feasibility_origin_rounded_and_model_info_structured(
    client: TestClient, pass_frame: tuple[int, str, str]
) -> None:
    f, _, receiver = pass_frame
    s = frame_state(MID, f)
    i = s.index(receiver)
    body = {
        "match_id": MID,
        "frame": f,
        "ensemble": 4,
        "interventions": [
            {"player_id": receiver, "x": float(s.xy[i, 0]) + 1, "y": float(s.xy[i, 1])}
        ],
    }
    j = client.post("/api/scenarios", json=body).json()
    ox = j["feasibility"][0]["origin_x"]
    assert ox is None or round(ox, 2) == ox
    info = j["observed"]["option_model"]["params"]
    assert info["n_train"] > 1000 and 0 < info["lomo_log_loss"] < 1 and info["n_bootstrap"] >= 16


def test_ensemble_samples_option_model_uncertainty(pass_frame: tuple[int, str, str]) -> None:
    """Option-model intervals must reflect bootstrap members, not only physics params."""
    from footymind.spatial.analysis import analyze
    from footymind.spatial.pitch_control import DEFAULT_PARAMS

    f, _, _ = pass_frame
    s = frame_state(MID, f)
    point = {
        o.receiver_id: o.p_success_model for o in analyze(s, DEFAULT_PARAMS, None).pass_options
    }
    members = [
        {o.receiver_id: o.p_success_model for o in analyze(s, DEFAULT_PARAMS, None, m).pass_options}
        for m in range(8)
    ]
    rid = next(iter(point))
    vals = [m[rid] for m in members]
    assert max(vals) - min(vals) > 1e-4  # same physics params, different coefficients


def test_two_players_moved_onto_same_spot_is_unsupported(pass_frame: tuple[int, str, str]) -> None:
    """Review finding #6: separation is checked on the final hypothetical state."""
    f, _, _ = pass_frame
    s = frame_state(MID, f)
    defs = [i for i, t in enumerate(s.teams) if t != s.attacking_team and not np.isnan(s.xy[i, 0])]
    a, b = defs[1], defs[2]
    mid = (s.xy[a] + s.xy[b]) / 2
    r = run_scenario(
        MID,
        f,
        [Intervention(s.player_ids[a], *mid), Intervention(s.player_ids[b], *mid)],
        ensemble=0,
    )
    assert all(fz.min_separation_m < 0.5 for fz in r.feasibility)
    assert not r.supported


def test_simulated_moves_are_feasibility_checked(pass_frame: tuple[int, str, str]) -> None:
    f, _, receiver = pass_frame
    s = frame_state(MID, f)
    i = s.index(receiver)
    r = run_scenario(
        MID,
        f,
        [Intervention(receiver, s.xy[i, 0] + 3.0, s.xy[i, 1])],
        policy="marker_follows",
        ensemble=0,
    )
    sim_moves = [m for m in r.movements if m.provenance.value == "simulated"]
    sim_feas = [fz for fz in r.feasibility if fz.kind == "simulated"]
    assert len(sim_feas) == len(sim_moves)
    assert {fz.player_id for fz in sim_feas} == {m.player_id for m in sim_moves}


def test_match_list_is_fast_and_does_not_load_tracking(client: TestClient) -> None:
    """Regression: summaries used to load every tracking bundle (~2.5 s with 23 matches)."""
    import time

    client.get("/api/matches")  # warm manifest cache
    t0 = time.perf_counter()
    r = client.get("/api/matches")
    assert r.status_code == 200 and len(r.json()) >= 1
    assert time.perf_counter() - t0 < 0.5


def test_period_time_starts_at_zero_each_period(client: TestClient) -> None:
    td = load_match(MID).tracking
    p2_first = int(td.frame[np.argmax(td.period == 2)])
    r = client.get(
        f"/api/matches/{MID}/replay", params={"start_frame": p2_first, "end_frame": p2_first + 50}
    ).json()
    assert r["period_time_s"][0] == pytest.approx(0.0)
    assert r["period_time_s"][-1] == pytest.approx(2.0, abs=0.05)
    a = client.get(f"/api/matches/{MID}/frames/{p2_first + 25}/analysis").json()
    assert a["period_time_s"] == pytest.approx(1.0, abs=0.05)


def test_frame_possession_resolves_deep_links(client: TestClient) -> None:
    j = client.get(f"/api/matches/{MID}/frames/53708/possession").json()
    assert j["possession_id"] == "metrica-sample-1:p0163"
    assert j["start_frame"] <= 53708 <= j["end_frame"]
    assert client.get(f"/api/matches/{MID}/frames/999999999/possession").status_code == 404


# --- opponent response policies (counterfactual/response.py) ---

POLICIES = [
    "none",
    "marker_follows",
    "press_carrier",
    "zonal_shift",
    "lane_cover",
    "learned",
    "learned_innovation",
]


def test_response_policies_endpoint(client: TestClient) -> None:
    r = client.get("/api/response-policies")
    assert r.status_code == 200
    body = r.json()
    assert [p["name"] for p in body] == POLICIES
    for p in body:
        assert set(p) >= {"name", "description", "assumptions", "validated", "validation_ref"}
        assert p["description"] and p["assumptions"]
        if p["name"] not in ("learned", "learned_innovation"):
            assert p["validated"] is False and p["validation_ref"] is None
    served = client.get("/api/response-policies", params={"match_id": MID}).json()
    for learned in [p for p in served if p["name"] in ("learned", "learned_innovation")]:
        if learned["validated"]:
            assert learned["validation_ref"] and learned["status"].startswith("validated")
        else:
            assert learned["status"].startswith("experimental")
    verdict = {p["name"]: p["verdict"] for p in served}
    assert verdict["none"] == "no_response"
    assert verdict["marker_follows"] == verdict["lane_cover"] == "observational_supported"
    assert verdict["press_carrier"] == verdict["zonal_shift"] == "not_supported"
    for learned in [p for p in served if p["name"] in ("learned", "learned_innovation")]:
        expected = {
            "learned": "forecast_validated",
            "learned_innovation": "observational_validated",
        }
        assert learned["verdict"] == (
            expected[learned["name"]] if learned["validated"] else "unchecked"
        )
    assert client.get("/api/response-policies", params={"match_id": "nope"}).status_code == 404


def _carrier_frame() -> tuple[int, str, str]:
    """A pass-start frame with an identified carrier, plus the receiver."""
    passes = load_match(MID).events.filter(pl.col("type") == "PASS")
    for row in passes.iter_rows(named=True):
        if row["start_frame"] < 5000:
            continue
        s = frame_state(MID, row["start_frame"])
        if s.carrier is not None and s.player_ids[s.carrier] == row["from_player_id"]:
            return row["start_frame"], row["from_player_id"], row["to_player_id"]
    raise AssertionError("no carrier frame")


@pytest.mark.parametrize("policy", POLICIES)
def test_scenario_with_each_policy(client: TestClient, policy: str) -> None:
    from footymind.counterfactual.response_model import load_response_model

    if policy == "learned" and load_response_model("metrica") is None:
        pytest.skip("learned response model not trained yet (footymind train-response-model)")
    f, carrier, receiver = _carrier_frame()
    s = frame_state(MID, f)
    who = receiver if policy in ("marker_follows", "lane_cover") else carrier
    i = s.index(who)
    body = {
        "match_id": MID,
        "frame": f,
        "ensemble": 0,
        "response_policy": policy,
        "interventions": [{"player_id": who, "x": float(s.xy[i, 0]) + 3, "y": float(s.xy[i, 1])}],
    }
    r = client.post("/api/scenarios", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["policy"]["name"] == policy
    assert isinstance(j["policy"]["validated"], bool)
    sim = [m for m in j["movements"] if m["provenance"] == "simulated"]
    for m in sim:
        k = s.index(m["player_id"])
        assert s.teams[k] != s.attacking_team  # only defenders respond
        assert m["player_id"] != who and m["reason"]
    simulated_feas = {x["player_id"] for x in j["feasibility"] if x["kind"] == "simulated"}
    assert simulated_feas == {m["player_id"] for m in sim}
    if policy == "none":
        assert sim == []
    if policy in ("press_carrier", "zonal_shift", "lane_cover", "learned"):
        assert sim or any("response policy" in w for w in j["warnings"])


def test_scenario_rejects_unknown_policy(client: TestClient) -> None:
    body = {
        "match_id": MID,
        "frame": 100,
        "response_policy": "teleport",
        "interventions": [{"player_id": "home_1", "x": 0.0, "y": 0.0}],
    }
    assert client.post("/api/scenarios", json=body).status_code == 422


def test_kinematic_feasibility_reported(pass_frame: tuple[int, str, str]) -> None:
    f, _, receiver = pass_frame
    s = frame_state(MID, f)
    i = s.index(receiver)
    x, y = float(s.xy[i, 0]), float(s.xy[i, 1])
    near = run_scenario(MID, f, [Intervention(receiver, x + 0.2, y)], ensemble=0)
    assert near.feasibility[0].kinematic_reachable in (True, False)
    far = run_scenario(MID, f, [Intervention(receiver, x + 9.0, y)], ensemble=0)
    assert far.feasibility[0].kinematic_reachable is False  # 9 m off its path in 1.5 s


SC = "skillcorner-1899585"


@pytest.mark.skipif(SC not in available_matches(), reason="SkillCorner data not available")
def test_learned_policy_on_broadcast_tracking(client: TestClient) -> None:
    from footymind.counterfactual.response_model import load_response_model

    if load_response_model("skillcorner") is None:
        pytest.skip("SkillCorner response model not trained yet")
    s = frame_state(SC, 39177)  # featured broadcast scenario (C27)
    assert s.carrier is not None
    c = s.player_ids[s.carrier]
    body = {
        "match_id": SC,
        "frame": 39177,
        "ensemble": 0,
        "response_policy": "learned",
        "interventions": [
            {"player_id": c, "x": float(s.xy[s.carrier, 0]) + 3, "y": float(s.xy[s.carrier, 1])}
        ],
    }
    j = client.post("/api/scenarios", json=body).json()
    assert j["policy"]["name"] == "learned"
    for m in j["movements"]:
        if m["provenance"] == "simulated":
            assert s.teams[s.index(m["player_id"])] != s.attacking_team
    pol = client.get("/api/response-policies", params={"match_id": SC}).json()[-1]
    assert pol["validated"] == j["policy"]["validated"]
