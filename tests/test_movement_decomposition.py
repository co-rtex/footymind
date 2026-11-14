"""Attack/defense movement decomposition on SYNTHETIC constructed geometry (not match data)."""

import numpy as np
import pytest

from footymind.ghost.movement_decomposition import _shapley, decompose
from tests.test_run_value import _state


def test_shapley_identity_and_additive_case() -> None:
    v = {"00": 0.4, "10": 0.6, "01": 0.3, "11": 0.5}  # additive: no interaction
    a, d, i = _shapley(v)
    assert a + d == pytest.approx(v["11"] - v["00"])
    assert a == pytest.approx(0.2) and d == pytest.approx(-0.1) and i == pytest.approx(0.0)


def test_nobody_moved_gives_zero() -> None:
    st = _state((25.0, 8.0))
    r = decompose(st, st.xy.copy(), np.zeros_like(st.xy))
    assert r.attack_effect == pytest.approx(0) and r.defense_effect == pytest.approx(0)
    assert r.interaction == pytest.approx(0)


def test_run_into_space_is_attack_credit_and_cover_is_defense_credit() -> None:
    st = _state((25.0, 8.0))
    # the runner came from behind the defender (blocked lane); defenders did not move
    earlier = st.xy.copy()
    earlier[1] = (25.0, 0.5)
    r = decompose(st, earlier, np.zeros_like(earlier))
    assert r.attack_effect > 0.02 and r.defense_effect == pytest.approx(0, abs=1e-9)
    # now the defender also moved across to block the new lane: negative defense effect
    st2 = _state((25.0, 8.0))
    st2.xy[3] = (15.0, 5.0)
    earlier2 = st2.xy.copy()
    earlier2[1] = (25.0, 0.5)
    earlier2[3] = (15.0, 0.0)
    r2 = decompose(st2, earlier2, np.zeros_like(earlier2))
    assert r2.defense_effect < 0
    fv = r2.v["forward"]
    assert r2.attack_effect + r2.defense_effect == pytest.approx(fv["11"] - fv["00"])


def test_carrier_never_moves_and_missing_history_counts_as_static() -> None:
    st = _state((25.0, 8.0))
    earlier = st.xy.copy()
    earlier[0] = (-20.0, 0.0)  # the carrier's history is ignored
    earlier[3] = np.nan  # defender with no earlier position: treated as not moved
    r = decompose(st, earlier, np.zeros_like(earlier))
    assert r.attack_effect == pytest.approx(0) and r.defense_effect == pytest.approx(0)
    assert r.n_defenders_moved == 1  # only the away GK had an earlier position


def test_best_option_saturates_on_safe_backward_pass() -> None:
    """Documents why the primary measure is the best *progressive* option."""
    st = _state((25.0, 8.0))
    earlier = st.xy.copy()
    earlier[1] = (25.0, 0.5)
    r = decompose(st, earlier, np.zeros_like(earlier))
    assert r.effects["any"].attack == pytest.approx(0, abs=1e-3)  # GK pass dominates
    assert r.effects["forward"].attack > 0.02


def test_rotation_placebo_keeps_displacement_size_and_attackers() -> None:
    from footymind.research.movement_decomposition_study import _rotate_defenders

    st = _state((25.0, 8.0))
    earlier = st.xy.copy()
    earlier[3] = (12.0, 4.0)  # a defender moved 5 m
    earlier[1] = (25.0, 0.5)
    xy, _ = _rotate_defenders(st, earlier, np.zeros_like(earlier), np.random.default_rng(1))
    assert np.linalg.norm(st.xy[3] - xy[3]) == pytest.approx(5.0)
    assert not np.allclose(xy[3], earlier[3])  # the direction changed
    assert np.allclose(xy[1], earlier[1])  # attackers untouched
