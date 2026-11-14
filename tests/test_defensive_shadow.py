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
    assert "experimental" in j["status"] and "not better than nearest-defender" in j["status"]
    assert j["provider"] == "metrica" and j["validation_claim"] == "C17"
    assert j["noise_floor"] == 0.025 and j["horizon_s"] == 1.0 and j["alt_radius_m"] == 3.5
    for d in j["defenders"]:
        if d["most_suppressed_receiver"] is not None:
            assert d["most_suppressed_delta"] > 0
    assert all(isinstance(d["distinguishable_from_noise"], bool) for d in j["defenders"])


def test_recovery_window_is_a_time_not_a_frame_count() -> None:
    # Regression: a 50-frame window meant 2 s on Metrica (25 Hz) but 5 s on SkillCorner (10 fps).
    from footymind.research.shadow_validation import recovery_window_frames

    assert recovery_window_frames(25.0) == 50
    assert recovery_window_frames(10.0) == 20


def test_shadow_status_is_per_provider_and_cites_the_ledger() -> None:
    """Regression (release review): one Metrica-based status was served for every provider."""
    from pathlib import Path

    from footymind.api.app import SHADOW_STATUS
    from footymind.paths import REPO_ROOT

    assert set(SHADOW_STATUS) == {"metrica", "skillcorner"}
    ledger = (REPO_ROOT / "docs" / "CLAIMS_LEDGER.md").read_text()
    for provider, (status, claim, ref) in SHADOW_STATUS.items():
        assert f"| {claim} |" in ledger and ref in ledger, provider
        assert (REPO_ROOT / ref).exists() and Path(ref).name == "results.json"
        assert claim in status and "Not a defender rating" in status
    assert "2x chance" in SHADOW_STATUS["metrica"][0]
    sc = SHADOW_STATUS["skillcorner"][0]
    assert "no better than chance" in sc and "2x" not in sc


SC = sorted(m for m in available_matches() if m.startswith("skillcorner-"))


@pytest.mark.skipif(not SC, reason="processed SkillCorner data not available")
def test_shadow_endpoint_on_skillcorner_cites_the_non_replication() -> None:
    import polars as pl

    from footymind.api.app import app
    from footymind.data.store import load_match

    c = TestClient(app)
    passes = load_match(SC[0]).events.filter(pl.col("type") == "PASS")
    for f in passes["start_frame"].to_list()[::25]:
        r = c.get(f"/api/matches/{SC[0]}/frames/{f}/defensive-shadow")
        if r.status_code == 200:
            break
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["provider"] == "skillcorner" and j["validation_claim"] == "C29"
    assert "no better than chance" in j["status"] and "2x" not in j["status"]
