"""Kinematic reachability envelope (synthetic inputs; no match data needed)."""

import numpy as np
import pytest

from footymind.counterfactual.kinematics import (
    DEFAULT_LIMITS,
    KinematicLimits,
    envelope_vertices,
    inside_polygon,
    prepare_velocity,
    project_onto_polygon,
    reachability_envelope,
)

LIM = KinematicLimits(reaction_s=0.25, max_accel_mps2=3.5, max_speed_mps=7.5)


def test_straight_line_from_rest() -> None:
    # 1 s of thrust after the delay: 0.5 * 3.5 * 1^2 = 1.75 m in any direction.
    e = reachability_envelope([0.0, 0.0], [0.0, 0.0], 1.25, LIM)
    assert e.contains([1.74, 0.0]) and not e.contains([1.76, 0.0])
    assert e.contains([0.0, -1.74]) and not e.contains([0.0, -1.80])
    np.testing.assert_allclose(e.clip([10.0, 0.0]), [1.75, 0.0], atol=1e-9)


def test_speed_limit_binds_for_a_sprinter() -> None:
    e = reachability_envelope([0.0, 0.0], [7.5, 0.0], 1.0, LIM)
    assert e.vertices[:, 0].max() == pytest.approx(7.5, abs=1e-6)
    assert e.contains([7.49, 0.0]) and not e.contains([7.6, 0.0])


def test_reversing_sprinter_cannot_get_back() -> None:
    # Running at 7 m/s: braking at 3.5 m/s^2 after a 0.25 s delay still leaves the player
    # 6.02 m ahead after 1 s. The constant-speed disc (7 m/s * 1 s) would accept (-3, 0).
    e = reachability_envelope([0.0, 0.0], [7.0, 0.0], 1.0, LIM)
    assert e.vertices[:, 0].min() == pytest.approx(7.0 - 0.5 * 3.5 * 0.75**2, abs=1e-6)
    assert not e.contains([0.0, 0.0])
    assert not e.contains([-3.0, 0.0])
    assert e.contains(e.coast_point)


def test_zero_horizon_only_the_start_is_reachable() -> None:
    e = reachability_envelope([1.0, 2.0], [3.0, -1.0], 0.0, LIM)
    assert e.contains([1.0, 2.0]) and not e.contains([1.01, 2.0])
    np.testing.assert_allclose(e.clip([30.0, 30.0]), [1.0, 2.0])


def test_within_reaction_delay_the_player_coasts() -> None:
    e = reachability_envelope([0.0, 0.0], [4.0, 0.0], 0.2, LIM)
    np.testing.assert_allclose(e.clip([0.0, 5.0]), [0.8, 0.0], atol=1e-9)


def test_nan_velocity_falls_back_to_rest_and_says_so() -> None:
    e = reachability_envelope([0.0, 0.0], [np.nan, np.nan], 1.25, LIM)
    rest = reachability_envelope([0.0, 0.0], [0.0, 0.0], 1.25, LIM)
    assert e.velocity_assumed and not rest.velocity_assumed
    assert any("velocity unavailable" in n for n in e.notes)
    np.testing.assert_allclose(e.vertices, rest.vertices)


def test_overspeed_is_capped_and_noted() -> None:
    e = reachability_envelope([0.0, 0.0], [11.0, 0.0], 1.0, LIM)
    assert e.velocity == pytest.approx((7.5, 0.0))
    assert any("capped" in n for n in e.notes)


def test_nan_position_rejected() -> None:
    with pytest.raises(ValueError):
        reachability_envelope([np.nan, 0.0], [0.0, 0.0], 1.0, LIM)


def test_projection_is_non_expansive_and_identity_inside() -> None:
    rng = np.random.default_rng(0)
    e = reachability_envelope([0.0, 0.0], [3.0, 2.0], 1.5, DEFAULT_LIMITS)
    for _ in range(200):
        a, b = rng.normal(0, 6, 2), rng.normal(0, 6, 2)
        ca, cb = e.clip(a), e.clip(b)
        assert np.linalg.norm(ca - cb) <= np.linalg.norm(a - b) + 1e-6
        assert e.contains(ca)
        if e.contains(a):
            np.testing.assert_allclose(ca, a)


def test_vectorised_matches_scalar_and_mirror_symmetric() -> None:
    rng = np.random.default_rng(1)
    p = rng.normal(0, 10, (5, 2))
    v, _, _ = prepare_velocity(rng.normal(0, 4, (5, 2)), LIM)
    verts = envelope_vertices(p, v, [0.5, 1.0], LIM)
    assert verts.shape == (2, 5, LIM.n_headings, 2)
    for i in range(5):
        e = reachability_envelope(p[i], v[i], 1.0, LIM)
        np.testing.assert_allclose(verts[1, i], e.vertices, atol=1e-12)
        q = rng.normal(p[i], 3)
        m = np.array([-1.0, 1.0])
        em = reachability_envelope(p[i] * m, v[i] * m, 1.0, LIM)
        assert em.contains(q * m) == e.contains(q)
        np.testing.assert_allclose(em.clip(q * m), e.clip(q) * m, atol=1e-9)
    assert inside_polygon(verts[0], verts[0][:, 0, :]).all()  # vertices count as inside
    # Zero thrust is always reachable. At the speed limit the coasting point lies on the
    # curved boundary, so the inscribed polygon may miss it by the chord sagitta (< 1 cm).
    coast = p + v * 1.0
    np.testing.assert_allclose(project_onto_polygon(verts[1], coast), coast, atol=1e-2)


def test_envelope_grows_with_horizon() -> None:
    small = reachability_envelope([0.0, 0.0], [2.0, 1.0], 0.6, LIM)
    big = reachability_envelope([0.0, 0.0], [2.0, 1.0], 1.2, LIM)

    # Not nested in general (the coasting point moves), but the area grows.
    def area(v: np.ndarray) -> float:
        x, y = v[:, 0], v[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    assert area(big.vertices) > area(small.vertices) > 0


def test_invalid_limits() -> None:
    with pytest.raises(ValueError):
        KinematicLimits(n_headings=7)
    with pytest.raises(ValueError):
        KinematicLimits(max_speed_mps=0.0)
