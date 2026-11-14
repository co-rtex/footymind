"""Off-ball opportunity under reactive defenders, on SYNTHETIC constructed geometry.

Every state here is constructed by hand (not match data). The match id "syn" has no option
model, so the physics heuristic is the opportunity model in these tests.
"""

import numpy as np
import pytest

from footymind.ghost.offball_opportunity import (
    ASSUMPTIONS,
    cv_background,
    eligible_attackers,
    offball_opportunity,
    rotate_history,
)
from footymind.spatial.state import FrameState
from footymind.tracking.model import Provenance


def _state(runner_now: tuple[float, float], marker_now: tuple[float, float]) -> FrameState:
    """SYNTHETIC: home_1 carrier at the origin, home_2 the runner, home_3 GK; away_1 a deep
    defender at (15, 0), away_2 the runner's marker, away_3 GK. Home attacks +x."""
    xy = np.array(
        [[0.0, 0.0], runner_now, [-45.0, 0.0], [15.0, 0.0], marker_now, [50.0, 0.0]], dtype=float
    )
    n = xy.shape[0]
    return FrameState(
        match_id="syn",
        frame=1,
        period=1,
        time_s=0.0,
        player_ids=["home_1", "home_2", "home_3", "away_1", "away_2", "away_3"],
        teams=["home", "home", "home", "away", "away", "away"],
        jerseys=[1, 2, 3, 1, 2, 3],
        xy=xy,
        vel=np.zeros((n, 2)),
        is_gk=np.array([False, False, True, False, False, True]),
        ball=np.array([0.0, 0.0]),
        attacking_team="home",
        possession_id="syn:p0",
        carrier=0,
        attack_sign=1,
        position_provenance=[Provenance.OBSERVED] * n,
        velocity_provenance=[Provenance.DERIVED] * n,
    )


def _runner(r: object) -> object:
    return next(a for a in r.attackers if a.player_id == "home_2")  # type: ignore[attr-defined]


def test_only_offball_outfield_attackers_and_estimated() -> None:
    st = _state((25.0, 8.0), (23.0, 6.0))
    assert eligible_attackers(st, st.xy.copy()) == [1]  # carrier and GK excluded
    r = offball_opportunity(st, st.xy.copy(), np.zeros_like(st.xy))
    assert r.provenance == Provenance.ESTIMATED
    assert all(a.provenance == Provenance.ESTIMATED for a in r.attackers)
    assert set(_runner(r).change) == set(ASSUMPTIONS)  # type: ignore[attr-defined]


def test_standing_still_is_zero_under_every_assumption() -> None:
    st = _state((25.0, 8.0), (23.0, 6.0))
    r = offball_opportunity(st, st.xy.copy(), np.zeros_like(st.xy))
    for k, ch in _runner(r).change.items():  # type: ignore[attr-defined]
        for m, v in ch.items():
            assert v == pytest.approx(0.0, abs=1e-12), (k, m)
    assert _runner(r).defenders_moved == {"marker_follows": 0, "lane_cover": 0}  # type: ignore[attr-defined]


def test_dragging_the_marker_adds_credit_under_marker_follows() -> None:
    """The runner came from in front of the deep defender, out wide; the marker followed.
    Holding the marker where it ended up (static) gives the counterfactual runner a free
    marker. Removing the response puts the marker back near the runner's earlier position, so
    the run is credited with the space it dragged the marker out of."""
    st = _state((25.0, 8.0), (23.0, 6.0))
    earlier = st.xy.copy()
    earlier[1] = (25.0, 0.5)
    vel0 = np.zeros_like(earlier)  # the run was unexpected: innovation = full displacement
    r = offball_opportunity(st, earlier, vel0)
    a = _runner(r)
    assert a.innovation_m == pytest.approx(7.5)  # type: ignore[attr-defined]
    assert a.defenders_moved["marker_follows"] == 1  # type: ignore[attr-defined]
    ch = a.change  # type: ignore[attr-defined]
    assert ch["marker_follows"]["own"] > ch["static"]["own"] + 0.01
    assert ch["lane_cover"]["own"] >= ch["static"]["own"] - 1e-9


def test_predictable_run_attributes_no_response_under_innovation() -> None:
    """If the runner's earlier velocity predicted the whole run, the innovation is zero: the
    policies attribute no response, so they equal static. The full attribution does not."""
    st = _state((25.0, 8.0), (23.0, 6.0))
    earlier = st.xy.copy()
    earlier[1] = (25.0, 0.5)
    vel0 = np.zeros_like(earlier)
    vel0[1] = (0.0, 7.5)  # 7.5 m/s for 1 s explains the displacement exactly
    inn = _runner(offball_opportunity(st, earlier, vel0))
    assert inn.innovation_m == pytest.approx(0.0, abs=1e-12)  # type: ignore[attr-defined]
    for k in ("marker_follows", "lane_cover"):
        for m in ("forward", "sum", "own"):
            assert inn.change[k][m] == pytest.approx(inn.change["static"][m]), (k, m)  # type: ignore[attr-defined]
    full = _runner(offball_opportunity(st, earlier, vel0, attribution="full"))
    assert full.change["marker_follows"]["own"] > full.change["static"]["own"] + 0.01  # type: ignore[attr-defined]


def test_unknown_earlier_velocity_leaves_policy_changes_unknown() -> None:
    st = _state((25.0, 8.0), (23.0, 6.0))
    earlier = st.xy.copy()
    earlier[1] = (25.0, 0.5)
    vel0 = np.zeros_like(earlier)
    vel0[1] = np.nan
    a = _runner(offball_opportunity(st, earlier, vel0))
    assert np.isnan(a.innovation_m)  # type: ignore[attr-defined]
    assert all(np.isnan(v) for v in a.change["marker_follows"].values())  # type: ignore[attr-defined]
    assert all(np.isnan(v) for v in a.change["lane_cover"].values())  # type: ignore[attr-defined]
    assert np.isfinite(a.change["static"]["own"])  # type: ignore[attr-defined]


def test_constant_velocity_background() -> None:
    st = _state((25.0, 8.0), (23.0, 6.0))
    earlier = st.xy.copy()
    earlier[4] = (20.0, 6.0)
    vel0 = np.zeros_like(earlier)
    vel0[4] = (1.0, 0.0)  # the marker was moving at 1 m/s: CV puts them at (21, 6)
    vel0[3] = np.nan  # the deep defender's earlier velocity is unknown: kept as observed
    bg, unresolved = cv_background(st, earlier, vel0, 1.0)
    assert bg.xy[4] == pytest.approx([21.0, 6.0]) and bg.xy[3] == pytest.approx([15.0, 0.0])
    assert unresolved == 1
    assert bg.xy[5] == pytest.approx([50.0, 0.0])  # the GK is never extrapolated
    assert bg.position_provenance[4] == Provenance.HYPOTHETICAL
    # with defenders exactly on their CV path, constant velocity equals static
    earlier2 = st.xy.copy()
    earlier2[1] = (25.0, 0.5)
    r = offball_opportunity(st, earlier2, np.zeros_like(earlier2))
    a = _runner(r)
    for m in ("forward", "sum", "own"):
        assert a.change["constant_velocity"][m] == pytest.approx(a.change["static"][m])  # type: ignore[attr-defined]


def test_rotation_placebo_keeps_sizes_and_defenders() -> None:
    st = _state((25.0, 8.0), (23.0, 6.0))
    earlier = st.xy.copy()
    earlier[1] = (25.0, 0.5)
    earlier[4] = (22.0, 1.0)
    vel0 = np.zeros_like(earlier)
    vel0[1] = (0.0, 3.0)
    rxy, rvel = rotate_history(st, earlier, vel0, [1], np.random.default_rng(1))
    assert np.linalg.norm(st.xy[1] - rxy[1]) == pytest.approx(7.5)
    assert np.linalg.norm(rvel[1]) == pytest.approx(3.0)
    inn = (st.xy[1] - earlier[1]) - vel0[1]
    rinn = (st.xy[1] - rxy[1]) - rvel[1]
    assert np.linalg.norm(rinn) == pytest.approx(np.linalg.norm(inn))
    assert rxy[4] == pytest.approx(earlier[4])  # defenders untouched


def test_rejects_bad_inputs() -> None:
    st = _state((25.0, 8.0), (23.0, 6.0))
    with pytest.raises(ValueError):
        offball_opportunity(st, st.xy.copy(), np.zeros_like(st.xy), attribution="half")
    st.carrier = None
    with pytest.raises(ValueError):
        offball_opportunity(st, st.xy.copy(), np.zeros_like(st.xy))
