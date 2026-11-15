"""Scenario engine with envelope v2 and learned_innovation on real data (skipped if absent)."""

import numpy as np
import polars as pl
import pytest

from footymind.counterfactual.kinematics import ENVELOPE_VERSION, KinematicLimitsV2, default_limits
from footymind.counterfactual.response import policy_info
from footymind.counterfactual.scenario import Intervention, run_scenario
from footymind.data.store import available_matches, load_match
from footymind.spatial.state import frame_state

pytestmark = pytest.mark.skipif(
    "metrica-sample-1" not in available_matches(), reason="processed Metrica data not available"
)
MID = "metrica-sample-1"


@pytest.fixture(scope="module")
def pass_frame() -> tuple[int, str, str]:
    p = load_match(MID).events.filter(pl.col("type") == "PASS").row(100, named=True)
    return p["start_frame"], p["from_player_id"], p["to_player_id"]


def test_learned_innovation_responds_and_is_simulated(pass_frame: tuple[int, str, str]) -> None:
    f, passer, _ = pass_frame
    s = frame_state(MID, f)
    i = s.index(passer)
    iv = Intervention(passer, float(s.xy[i, 0]) + 3.0, float(s.xy[i, 1]))
    r = run_scenario(MID, f, [iv], policy="learned_innovation", ensemble=0)
    sim = [m for m in r.movements if m.provenance.value == "simulated"]
    assert sim, "a 3 m relocation should move at least one defender"
    assert max(np.hypot(m.to_xy[0] - m.from_xy[0], m.to_xy[1] - m.from_xy[1]) for m in sim) > 0.1
    for m in sim:
        assert s.teams[s.index(m.player_id)] != s.attacking_team
    assert r.policy.name == "learned_innovation"
    assert r.policy.validated == policy_info("learned_innovation", "metrica").validated


def test_kinematic_diagnostic_uses_the_envelope_in_force(pass_frame: tuple[int, str, str]) -> None:
    f, _, receiver = pass_frame
    s = frame_state(MID, f)
    i = s.index(receiver)
    r = run_scenario(
        MID, f, [Intervention(receiver, float(s.xy[i, 0]), float(s.xy[i, 1]))], ensemble=0
    )
    # the observed position itself is (almost surely) inside the envelope from 1.5 s earlier
    assert r.feasibility[0].kinematic_reachable is True
    if ENVELOPE_VERSION == 2:
        assert isinstance(default_limits("metrica"), KinematicLimitsV2)
