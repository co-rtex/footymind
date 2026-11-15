"""Evidence-grounded explanations: SYNTHETIC constructed frames plus one real-data check."""

import numpy as np
import pytest

from footymind.data.store import available_matches
from footymind.spatial.analysis import analyze
from footymind.spatial.explain import explain
from footymind.spatial.state import frame_state
from footymind.tracking.model import Provenance
from tests.test_run_value import _state


def test_statements_are_grounded_in_metrics() -> None:
    a = analyze(_state((25.0, 8.0)))
    stm = explain(a)
    kinds = {s.kind for s in stm}
    assert {"pressure", "option", "caveat"} <= kinds and "trend" not in kinds
    pres = next(s for s in stm if s.kind == "pressure")
    assert pres.value == pytest.approx(round(a.pressure.nearest_defender_distance_m, 2))
    assert f"{a.pressure.nearest_defender_distance_m:.1f} m" in pres.text
    for s in stm:
        if s.kind == "option" and s.metric.startswith("pass_options.p_success"):
            assert s.provenance == Provenance.ESTIMATED
            assert set(s.players) <= set(a.state.player_ids)


def test_trend_statements_need_finite_temporal_values() -> None:
    a = analyze(_state((25.0, 8.0)))
    t = {
        "passer_control_s": 1.24,
        "press_nearest_def_change_m_l10": -2.5,
        "open_lanes_change_l10": 1.0,
    }
    trends = [s for s in explain(a, t) if s.kind == "trend"]
    assert [s.metric for s in trends] == [
        "temporal.passer_control_s",
        "temporal.press_nearest_def_change_m_l10",
        "temporal.open_lanes_change_l10",
    ]
    assert "shrank by 2.5 m" in trends[1].text and "building" in trends[1].text
    nan_t = {k: float("nan") for k in t}
    assert not [s for s in explain(a, nan_t) if s.kind == "trend"]  # no metric, no claim
    small = {"press_nearest_def_change_m_l10": 0.4, "open_lanes_change_l10": 0.0}
    assert not [s for s in explain(a, small) if s.kind == "trend"]


def test_no_ball_means_no_option_claims() -> None:
    st = _state((25.0, 8.0))
    st.ball = np.array([np.nan, np.nan])
    stm = explain(analyze(st))
    assert not [s for s in stm if s.kind == "option"]
    assert any("ball position not observed" in s.text.lower() for s in stm if s.kind == "caveat")


MET = [m for m in available_matches() if m.startswith("metrica-")]


@pytest.mark.skipif(not MET, reason="Metrica data not ingested")
def test_real_frame_references_only_players_in_the_frame() -> None:
    st = frame_state(MET[0], 3000)
    stm = explain(analyze(st))
    assert stm and all(s.text.endswith(".") for s in stm)
    for s in stm:
        assert set(s.players) <= set(st.player_ids)
        assert s.to_dict()["provenance"] in {p.value for p in Provenance}


def test_statements_about_extrapolated_players_are_estimated() -> None:
    st = _state((25.0, 8.0))
    st.position_provenance[0] = Provenance.ESTIMATED  # the carrier is off camera
    t = {"passer_control_s": 12.0}
    stm = explain(analyze(st), t)
    pres = next(s for s in stm if s.kind == "pressure")
    tob = next(s for s in stm if s.metric == "temporal.passer_control_s")
    assert pres.provenance == Provenance.ESTIMATED and tob.provenance == Provenance.ESTIMATED
    assert "at least" in tob.text  # capped time-on-ball is a lower bound
