"""Kinematic envelope v2 and the v1 regression (synthetic inputs; no match data needed)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from footymind.counterfactual import kinematics as kin
from footymind.counterfactual.kinematics import (
    DEFAULT_LIMITS,
    KinematicLimitsV2,
    default_limits,
    envelope_polygons,
    envelope_vertices,
    inside_polygon,
    prepare_velocity,
    reachability_envelope,
)

V2 = KinematicLimitsV2(position_sigma_m=0.03, velocity_sigma_mps=0.08)
MIRRORS = (np.array([-1.0, 1.0]), np.array([1.0, -1.0]), np.array([-1.0, -1.0]))


def _area(v: np.ndarray) -> float:
    x, y = v[..., 0], v[..., 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _convex(v: np.ndarray) -> bool:
    a = np.roll(v, -1, axis=0) - v
    b = np.roll(v, -2, axis=0) - np.roll(v, -1, axis=0)
    return bool(np.all(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] >= -1e-9))


def test_v1_regression_bit_for_bit() -> None:
    # Values recorded from the session-3 implementation before v2 was added.
    p = np.array([[0.0, 0.0], [10.0, -5.0], [-20.0, 3.0]])
    v = np.array([[0.0, 0.0], [6.0, 2.0], [-3.0, -7.0]])
    vp, _, _ = prepare_velocity(v, DEFAULT_LIMITS)
    verts = envelope_vertices(p, vp, [0.2, 0.5, 1.0, 1.5])
    expected = [
        [0.0, 316.8, -910.540631659],
        [-0.0, 432.0, -1050.0905779987],
        [-0.0, 619.823535218, -1267.2242183056],
        [-0.0, 790.1654704865, -1462.5004740867],
    ]
    np.testing.assert_allclose(verts.sum(axis=(-1, -2)), expected, atol=1e-8)
    np.testing.assert_allclose(
        verts[2, 1, ::12],
        [
            [16.7231435005, -3.0675360997],
            [15.9988593087, -2.0164936598],
            [15.015625, -3.0],
            [16.0, -3.984375],
        ],
        atol=1e-9,
    )
    # The version-generic entry point is the same computation for v1.
    poly, assumed, _ = envelope_polygons(p, v, [0.2, 0.5, 1.0, 1.5], DEFAULT_LIMITS)
    np.testing.assert_array_equal(poly, verts)
    assert not assumed.any()


def test_default_limits_selection() -> None:
    assert default_limits(version=1) is DEFAULT_LIMITS
    m = default_limits("metrica", version=2)
    assert isinstance(m, KinematicLimitsV2)
    assert (m.position_sigma_m, m.velocity_sigma_mps) == kin.PROVIDER_NOISE["metrica"]
    bare = default_limits("unknown", version=2)
    assert isinstance(bare, KinematicLimitsV2) and bare.margin_m(1.0) == 0.0
    assert "noise margin" not in bare.describe()
    with pytest.raises(ValueError):
        default_limits(version=3)
    current = default_limits("skillcorner")
    assert current.version == kin.ENVELOPE_VERSION


def test_invalid_v2_limits() -> None:
    with pytest.raises(ValueError):
        KinematicLimitsV2(max_accel_mps2=-1.0)
    with pytest.raises(ValueError):
        KinematicLimitsV2(max_speed_mps=0.0)
    with pytest.raises(ValueError):
        KinematicLimitsV2(position_sigma_m=-0.1)
    with pytest.raises(ValueError):
        KinematicLimitsV2(n_headings=9)


def test_straight_sprint_from_rest_matches_force_velocity_solution() -> None:
    # Along the run a = A0 (1 - v / V0): x(t) = V0 (t - tau (1 - exp(-t / tau))), tau = V0 / A0.
    lim = KinematicLimitsV2(velocity_lag_s=0.0)
    e = reachability_envelope([0.0, 0.0], [0.0, 0.0], 1.0, lim)
    tau = lim.max_speed_mps / lim.propulsive_accel_mps2
    x = lim.max_speed_mps * (1.0 - tau * (1.0 - math.exp(-1.0 / tau)))
    r = np.linalg.norm(e.vertices, axis=1)
    assert r.min() == pytest.approx(x, rel=0.01)  # explicit integration at dt = 0.02 s
    assert r.max() <= x / math.cos(math.pi / lim.n_headings) * 1.01


def test_braking_and_turning_differ_from_speeding_up() -> None:
    lim = KinematicLimitsV2(velocity_lag_s=0.0)
    e = reachability_envelope([0.0, 0.0], [7.0, 0.0], 1.0, lim)
    ahead = e.vertices[:, 0].max() - 7.0  # extra distance by speeding up
    behind = 7.0 - e.vertices[:, 0].min()  # distance given up by braking
    assert behind == pytest.approx(0.5 * lim.max_accel_mps2, rel=0.02)
    assert ahead < 0.5 * lim.propulsive_accel_mps2 * (1 - 7.0 / 9.0) * 1.05
    iso = reachability_envelope(
        [0.0, 0.0], [7.0, 0.0], 1.0, KinematicLimitsV2(velocity_lag_s=0.0, force_velocity=False)
    )
    assert iso.vertices[:, 0].max() > e.vertices[:, 0].max()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_v2_convex_contains_coast_point_and_grows_with_horizon(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(20):
        p, v = rng.normal(0, 15, 2), rng.normal(0, 3, 2)
        areas = []
        for t in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
            e = reachability_envelope(p, v, t, V2)
            assert _convex(e.vertices)
            assert e.contains(e.coast_point)
            areas.append(_area(e.vertices))
        assert all(b > a for a, b in itertools.pairwise(areas))


def test_v2_nested_in_horizon_from_rest() -> None:
    small = reachability_envelope([0.0, 0.0], [0.0, 0.0], 0.5, V2)
    big = reachability_envelope([0.0, 0.0], [0.0, 0.0], 1.0, V2)
    assert inside_polygon(big.vertices, small.vertices).all()


def test_v2_mirror_invariance() -> None:
    rng = np.random.default_rng(3)
    for _ in range(30):
        p, v = rng.normal(0, 10, 2), rng.normal(0, 4, 2)
        e = reachability_envelope(p, v, 1.0, V2)
        for m in MIRRORS:
            em = reachability_envelope(p * m, v * m, 1.0, V2)
            for _ in range(5):
                q = rng.normal(e.coast_point, 3)
                assert em.contains(q * m) == e.contains(q)
                np.testing.assert_allclose(em.clip(q * m), e.clip(q) * m, atol=1e-9)


def test_v2_nan_velocity_is_the_speed_disc_not_rest() -> None:
    e = reachability_envelope([0.0, 0.0], [np.nan, np.nan], 1.0, V2)
    assert e.velocity_assumed
    assert any("velocity unavailable" in n for n in e.notes)
    support = np.max(e.vertices @ kin._headings(V2.n_headings).T, axis=0)
    np.testing.assert_allclose(support, V2.max_speed_mps * 1.0 + V2.margin_m(1.0), rtol=1e-9)
    # It contains every envelope with a known velocity up to the speed cap.
    rng = np.random.default_rng(4)
    for _ in range(20):
        ang = rng.uniform(0, 2 * np.pi)
        v = rng.uniform(0, V2.max_speed_mps) * np.array([np.cos(ang), np.sin(ang)])
        known = reachability_envelope([0.0, 0.0], v, 1.0, V2)
        assert inside_polygon(e.vertices, known.vertices).all()
    # Vectorised: NaN rows do not affect other rows.
    poly, assumed, _ = envelope_polygons(
        np.zeros((2, 2)), np.array([[np.nan, np.nan], [2.0, 0.0]]), [1.0], V2
    )
    assert assumed.tolist() == [True, False]
    single = reachability_envelope([0.0, 0.0], [2.0, 0.0], 1.0, V2)
    np.testing.assert_allclose(poly[0, 1], single.vertices, atol=1e-12)


def test_v2_margin_offsets_the_support_exactly() -> None:
    bare = KinematicLimitsV2()
    noisy = KinematicLimitsV2(position_sigma_m=0.05, velocity_sigma_mps=0.2)
    assert noisy.margin_m(1.0) == pytest.approx(2.45 * math.sqrt(2 * 0.05**2 + 0.2**2))
    a = reachability_envelope([1.0, 1.0], [3.0, -2.0], 1.0, bare).vertices
    b = reachability_envelope([1.0, 1.0], [3.0, -2.0], 1.0, noisy).vertices
    u = kin._headings(bare.n_headings)
    sa = np.max(a @ u.T, axis=0)
    sb = np.max(b @ u.T, axis=0)
    np.testing.assert_allclose(sb - sa, noisy.margin_m(1.0), atol=1e-9)


def test_v2_projection_non_expansive() -> None:
    rng = np.random.default_rng(5)
    e = reachability_envelope([0.0, 0.0], [4.0, 1.0], 1.5, V2)
    for _ in range(200):
        a, b = rng.normal(0, 8, 2), rng.normal(0, 8, 2)
        ca, cb = e.clip(a), e.clip(b)
        assert np.linalg.norm(ca - cb) <= np.linalg.norm(a - b) + 1e-6
        assert e.contains(ca)


def test_v2_vectorised_matches_scalar_and_describes_itself() -> None:
    rng = np.random.default_rng(6)
    p, v = rng.normal(0, 10, (4, 2)), rng.normal(0, 3, (4, 2))
    poly, _, _ = envelope_polygons(p, v, [0.5, 1.0], V2)
    assert poly.shape == (2, 4, V2.n_headings, 2)
    for i in range(4):
        e = reachability_envelope(p[i], v[i], 1.0, V2)
        np.testing.assert_allclose(poly[1, i], e.vertices, atol=1e-12)
    assert "envelope v2" in V2.describe() and "noise margin" in V2.describe()
    assert V2.reaction_s == 0.0 and V2.version == 2 and DEFAULT_LIMITS.version == 1
