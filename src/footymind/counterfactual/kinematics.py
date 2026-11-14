"""Kinematic reachability envelopes for simulated player responses.

The question answered here is purely physical: given a player's position ``p0`` and velocity
``v0`` at some instant, which points could they occupy ``T`` seconds later? It is **not** a
statement about what the player would choose to do. A reachable move is not a tactically
plausible one.

Motion model (a bounded double integrator)
------------------------------------------
1. Reaction delay ``reaction_s``: the player keeps the initial velocity unchanged
   (``x = p0 + v0 t``). A player sprinting one way cannot start turning immediately.
2. Afterwards the acceleration magnitude is at most ``max_accel_mps2`` in any direction
   (speeding up, braking and turning share the same bound) and the speed is at most
   ``max_speed_mps``.

The system is linear with convex constraints, so the true reachable set is convex. It is
approximated from the inside: for ``n_headings`` equally spaced headings the player thrusts
at full acceleration in that fixed heading, with the speed projected back onto the speed
limit (a projection never changes the velocity by more than the thrust, so every simulated
path respects both bounds). The endpoints are reachable, so their convex polygon is an
under-approximation of the reachable set. Without the speed limit it is exact up to the
polygon discretisation: a disc of radius ``a (T - reaction)^2 / 2`` around the coasting point
``p0 + v0 T``.

Constants (a priori, not fitted; ``research.response_validation`` reports how often real
defender positions fall inside the envelope):

* ``REACTION_S = 0.25``: simple visual reaction times of trained athletes are around
  0.2 s; 0.25 s adds a little for recognising the change. The pitch-control model's 0.7 s
  (Spearman 2018) also includes the decision to go for the ball, which is a different
  question.
* ``MAX_ACCEL_MPS2 = 3.5``: GPS studies classify accelerations above about 3 m/s^2 as high
  intensity (e.g. Varley & Aughey 2013, >2.78 m/s^2). One bound is used for speeding up,
  braking and turning. Real braking peaks are higher, so the envelope is deliberately tight.
* ``MAX_SPEED_MPS = 7.5``: 27 km/h, above the 25.2 km/h sprint threshold used in match
  analysis (e.g. Bradley et al. 2009) and below elite peak sprint speeds (about 9-10 m/s).
  It is a typical-player bound, not a peak.

Unknown velocity (NaN): the envelope falls back to a start from rest and says so
(``velocity_assumed``). A start from rest is the symmetric choice: it understates the reach
in the direction of an unseen run and overstates it against that run. It is still far
tighter than the constant-speed disc used by the scenario feasibility check.

Envelope v2 (session 4, ``KinematicLimitsV2``)
---------------------------------------------
v1 above is kept bit-for-bit (``KinematicLimits``, ``DEFAULT_LIMITS``). v2 changes the model
for reasons stated before its coverage was evaluated (``docs/research/simulation_session4.md``,
``research/envelope_validation.py``):

1. *No reaction delay.* The current acceleration of a player is not observed. Holding the
   velocity for 0.25 s assumes it is zero; with an unknown acceleration bounded like any
   other, the delay does not restrict the reachable set at all. (A delay restricts how fast a
   player can respond to a *new* stimulus, which a set of reachable points cannot express.)
2. *Acceleration by direction.* Braking and turning share a friction/strength bound
   ``max_accel_mps2 = 7`` m/s^2 (the maximum player acceleration of Spearman's pitch-control
   model and its reference implementation; peak decelerations in team sports reach 6-9 m/s^2,
   Harper et al. 2019). Speeding up along the direction of motion is also limited by a linear
   force-velocity profile, ``a <= propulsive_accel_mps2 * (1 - speed / max_speed_mps)`` with
   7 m/s^2 and 9 m/s (sprint force-velocity profiling of footballers, Samozino-Morin method:
   horizontal F0 about 7-8 N/kg, V0 about 9 m/s). v1's 3.5 m/s^2 is a *high-intensity
   threshold* from GPS studies, which real players exceed routinely; it is not a maximum.
3. *Velocity lag.* The causal 0.2 s velocity is the mean velocity over the last 0.2 s, i.e.
   the velocity about ``velocity_lag_s = 0.1`` s ago. The envelope therefore starts at
   ``p0 - v0 * lag`` with velocity ``v0`` and runs for ``T + lag``: the coasting point is
   unchanged and the acceleration acts for the extra 0.1 s the estimate is out of date.
4. *Measurement margin.* Positions and the velocity estimate are noisy. The polygon is grown
   by ``margin_z * sqrt(2 sigma_p^2 + (sigma_v T)^2)`` with ``margin_z = 2.45`` (the 95%
   radius of an isotropic 2-D Gaussian) and per-provider noise estimated from the jitter of
   near-stationary players (``research/envelope_noise.py``), never from coverage.
5. *Unknown velocity* is not turned into a start from rest: the envelope is the disc of
   radius ``max_speed_mps * T`` (plus the margin), which contains every admissible path.

Geometry. Constant-heading strategies give reachable endpoints. v2 represents the set by its
support function in the ``n_headings`` directions (the largest endpoint projection plus the
margin) and intersects those half-planes. The polygon is convex by construction (so
projection onto it stays non-expansive) and contains every simulated endpoint; it can exceed
the convex hull of the endpoints by at most ``(1 / cos(pi / n) - 1)`` of the radius (0.2% at
48 headings).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

F = npt.NDArray[np.float64]

REACTION_S = 0.25
MAX_ACCEL_MPS2 = 3.5
MAX_SPEED_MPS = 7.5
N_HEADINGS = 48  # even, so the heading set is symmetric under x -> -x and y -> -y
DT_S = 0.02
_TOL = 1e-7  # metres; points this close to the boundary count as inside


@dataclass(frozen=True)
class KinematicLimits:
    reaction_s: float = REACTION_S
    max_accel_mps2: float = MAX_ACCEL_MPS2
    max_speed_mps: float = MAX_SPEED_MPS
    n_headings: int = N_HEADINGS
    dt_s: float = DT_S

    def __post_init__(self) -> None:
        if self.reaction_s < 0 or self.max_accel_mps2 < 0 or self.max_speed_mps <= 0:
            raise ValueError("kinematic limits must be non-negative (max speed positive)")
        if self.n_headings < 8 or self.n_headings % 2:
            raise ValueError("n_headings must be an even number >= 8")
        if self.dt_s <= 0:
            raise ValueError("dt_s must be positive")

    @property
    def version(self) -> int:
        return 1

    def describe(self) -> str:
        return (
            f"{self.reaction_s:.2f} s reaction, <= {self.max_accel_mps2:.1f} m/s^2, "
            f"<= {self.max_speed_mps:.1f} m/s"
        )


DEFAULT_LIMITS = KinematicLimits()

# Envelope v2 constants (a priori; see the module docstring).
V2_MAX_ACCEL_MPS2 = 7.0
V2_PROPULSIVE_ACCEL_MPS2 = 7.0
V2_MAX_SPEED_MPS = 9.0
V2_VELOCITY_LAG_S = 0.1
V2_MARGIN_Z = 2.45  # sqrt of the 95% quantile of chi^2 with 2 degrees of freedom


@dataclass(frozen=True)
class KinematicLimitsV2:
    max_accel_mps2: float = V2_MAX_ACCEL_MPS2  # braking, turning, any direction
    propulsive_accel_mps2: float = V2_PROPULSIVE_ACCEL_MPS2  # along the run, at rest
    max_speed_mps: float = V2_MAX_SPEED_MPS  # speed cap and force-velocity intercept
    velocity_lag_s: float = V2_VELOCITY_LAG_S
    position_sigma_m: float = 0.0
    velocity_sigma_mps: float = 0.0
    margin_z: float = V2_MARGIN_Z
    force_velocity: bool = True  # False: isotropic bound (ablation only)
    n_headings: int = N_HEADINGS
    dt_s: float = DT_S
    noise_source: str = "none"

    def __post_init__(self) -> None:
        if min(self.max_accel_mps2, self.propulsive_accel_mps2, self.velocity_lag_s) < 0:
            raise ValueError("kinematic limits must be non-negative")
        if self.max_speed_mps <= 0:
            raise ValueError("max speed must be positive")
        if min(self.position_sigma_m, self.velocity_sigma_mps, self.margin_z) < 0:
            raise ValueError("noise parameters must be non-negative")
        if self.n_headings < 8 or self.n_headings % 2:
            raise ValueError("n_headings must be an even number >= 8")
        if self.dt_s <= 0:
            raise ValueError("dt_s must be positive")

    @property
    def version(self) -> int:
        return 2

    @property
    def reaction_s(self) -> float:
        return 0.0

    def margin_m(self, t: float) -> float:
        """Measurement-error margin added to the envelope at horizon ``t`` (seconds)."""
        var = 2.0 * self.position_sigma_m**2 + (self.velocity_sigma_mps * t) ** 2
        return float(self.margin_z * math.sqrt(var))

    def describe(self) -> str:
        fv = (
            f", speeding up <= {self.propulsive_accel_mps2:.0f}(1 - v/{self.max_speed_mps:.0f}) "
            "m/s^2"
            if self.force_velocity
            else ""
        )
        noise = (
            f", noise margin {self.margin_z:.2f} sigma (position {self.position_sigma_m:.3f} m, "
            f"velocity {self.velocity_sigma_mps:.2f} m/s)"
            if self.margin_z > 0 and (self.position_sigma_m > 0 or self.velocity_sigma_mps > 0)
            else ""
        )
        return (
            f"envelope v2: no reaction delay, <= {self.max_accel_mps2:.1f} m/s^2 braking or "
            f"turning{fv}, <= {self.max_speed_mps:.1f} m/s, {self.velocity_lag_s:.2f} s "
            f"velocity lag{noise}"
        )


Limits = KinematicLimits | KinematicLimitsV2

# Which envelope the response policies and the scenario feasibility diagnostic use.
# 1 = the session-3 a priori envelope; 2 = envelope v2 with the provider's noise margin.
ENVELOPE_VERSION = 1
# Per-axis tracking noise from near-stationary players (research/envelope_noise.py),
# pooled over every match of the provider; see experiments/results/envelope_validation_v1*.
# (sigma_p in m, sigma_v in m/s); estimated before any v2 coverage was computed.
PROVIDER_NOISE: dict[str, tuple[float, float]] = {
    "metrica": (0.0173, 0.0686),
    "skillcorner": (0.0326, 0.0795),
}


def default_limits(provider: str | None = None, version: int | None = None) -> Limits:
    """The envelope limits in force (``ENVELOPE_VERSION`` unless ``version`` is given).

    v2 carries the provider's measured noise margin; for an unknown provider it has none
    (pure kinematics), and ``describe()`` says so by omission.
    """
    ver = ENVELOPE_VERSION if version is None else version
    if ver == 1:
        return DEFAULT_LIMITS
    if ver != 2:
        raise ValueError(f"unknown envelope version {ver}")
    if provider in PROVIDER_NOISE:
        sp, sv = PROVIDER_NOISE[provider]
        return KinematicLimitsV2(
            position_sigma_m=sp, velocity_sigma_mps=sv, noise_source=f"{provider} tracking"
        )
    return KinematicLimitsV2()


def prepare_velocity(
    v0: npt.ArrayLike, limits: KinematicLimits
) -> tuple[F, npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Velocities used by the envelope: NaN -> rest, speed capped at the limit.

    Returns ``(v, assumed, capped)``; the last two are boolean masks over the leading shape.
    """
    v = np.array(v0, dtype=np.float64)
    assumed = np.isnan(v).any(axis=-1)
    v[assumed] = 0.0
    s = np.linalg.norm(v, axis=-1)
    capped = s > limits.max_speed_mps
    scale = np.where(capped, limits.max_speed_mps / np.maximum(s, 1e-12), 1.0)
    return v * scale[..., None], assumed, capped


def envelope_vertices(
    p0: npt.ArrayLike,
    v0: npt.ArrayLike,
    times_s: Sequence[float],
    limits: KinematicLimits = DEFAULT_LIMITS,
) -> F:
    """Envelope polygons for every time in ``times_s``.

    ``p0`` and ``v0`` have shape ``(..., 2)``; ``v0`` must already be prepared
    (:func:`prepare_velocity`). The result has shape ``(len(times_s), ..., n_headings, 2)``,
    vertices ordered by heading. Each vertex is the position reached at that time by a
    constant-heading, full-thrust strategy, which is also the endpoint of that strategy at
    every earlier time, so one integration serves all horizons.
    """
    p = np.asarray(p0, dtype=np.float64)
    v_init = np.asarray(v0, dtype=np.float64)
    times = [float(t) for t in times_s]
    if any(t < 0 for t in times):
        raise ValueError("times must be non-negative")
    n = limits.n_headings
    theta = 2.0 * np.pi * np.arange(n) / n
    thrust = limits.max_accel_mps2 * np.stack([np.cos(theta), np.sin(theta)], axis=-1)  # (n, 2)
    out = np.empty((len(times), *p.shape[:-1], n, 2), dtype=np.float64)
    tau = limits.reaction_s
    # Before thrust starts the motion is linear (degenerate polygon).
    for k, t in enumerate(times):
        if t <= tau:
            out[k] = (p + v_init * t)[..., None, :]
    later = sorted({t for t in times if t > tau})
    if not later:
        return out
    # Integration grid: regular steps after the reaction delay, plus every requested time.
    t_end = later[-1]
    grid = [tau + i * limits.dt_s for i in range(1, math.ceil((t_end - tau) / limits.dt_s))]
    grid = sorted({*grid, *later})
    x = np.broadcast_to((p + v_init * tau)[..., None, :], (*p.shape[:-1], n, 2)).copy()
    v = np.broadcast_to(v_init[..., None, :], x.shape).copy()
    vmax = limits.max_speed_mps
    t_prev = tau
    reached: dict[float, F] = {}
    for t in grid:
        h = t - t_prev
        vn = v + thrust * h
        s = np.linalg.norm(vn, axis=-1, keepdims=True)
        vn = vn * np.minimum(1.0, vmax / np.maximum(s, 1e-12))
        x = x + 0.5 * (v + vn) * h
        v = vn
        t_prev = t
        if t in later:
            reached[t] = x.copy()
    for k, t in enumerate(times):
        if t > tau:
            out[k] = reached[t]
    return out


def _headings(n: int) -> F:
    theta = 2.0 * np.pi * np.arange(n) / n
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)  # (n, 2)


def support_polygon(h: F, n: int) -> F:
    """Vertices of the polygon ``{x : x . u_k <= h_k}`` for the ``n`` heading normals.

    ``h`` has shape ``(..., n)``. When ``h`` is the support function of a convex set in those
    directions every half-plane is tangent, and vertex ``k`` is the intersection of the
    boundary lines ``k`` and ``k + 1`` (counter-clockwise order).
    """
    u = _headings(n)
    u1 = np.roll(u, -1, axis=0)
    h1 = np.roll(h, -1, axis=-1)
    det = math.sin(2.0 * math.pi / n)
    wx = (h * u1[:, 1] - h1 * u[:, 1]) / det
    wy = (h1 * u[:, 0] - h * u1[:, 0]) / det
    return np.stack([wx, wy], axis=-1)


def _v2_endpoints(p: F, v: F, run_times: list[float], lim: KinematicLimitsV2) -> dict[float, F]:
    """Constant-heading endpoints ``(..., n, 2)`` after each time in ``run_times`` (> 0)."""
    n = lim.n_headings
    u = _headings(n)
    x = np.broadcast_to(p[..., None, :], (*p.shape[:-1], n, 2)).copy()
    vel = np.broadcast_to(v[..., None, :], x.shape).copy()
    t_end = max(run_times)
    grid = sorted({*(i * lim.dt_s for i in range(1, math.ceil(t_end / lim.dt_s))), *run_times})
    out: dict[float, F] = {}
    t_prev = 0.0
    a_max, vmax = lim.max_accel_mps2, lim.max_speed_mps
    for t in grid:
        h = t - t_prev
        if lim.force_velocity:
            s = np.linalg.norm(vel, axis=-1)
            c = np.sum(vel * u, axis=-1) / np.maximum(s, 1e-12)  # cos(thrust, run)
            a_fwd = lim.propulsive_accel_mps2 * np.maximum(0.0, 1.0 - s / vmax)
            mag = np.where(c > 1e-12, np.minimum(a_max, a_fwd / np.maximum(c, 1e-12)), a_max)
        else:
            mag = np.full(x.shape[:-1], a_max)
        vn = vel + mag[..., None] * u * h
        sn = np.linalg.norm(vn, axis=-1, keepdims=True)
        vn = vn * np.minimum(1.0, vmax / np.maximum(sn, 1e-12))
        x = x + 0.5 * (vel + vn) * h
        vel = vn
        t_prev = t
        if t in run_times:
            out[t] = x.copy()
    return out


def envelope_polygons(
    p0: npt.ArrayLike,
    v0: npt.ArrayLike,
    times_s: Sequence[float],
    limits: Limits = DEFAULT_LIMITS,
) -> tuple[F, npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Envelope polygons for raw (possibly NaN) velocities, for either envelope version.

    Returns ``(vertices, velocity_assumed, speed_capped)``; vertices have shape
    ``(len(times_s), ..., n_headings, 2)``. For v1 this is exactly
    ``envelope_vertices(p0, prepare_velocity(v0)[0], ...)``.
    """
    if isinstance(limits, KinematicLimits):
        v, assumed, capped = prepare_velocity(v0, limits)
        return envelope_vertices(p0, v, times_s, limits), assumed, capped
    p = np.asarray(p0, dtype=np.float64)
    times = [float(t) for t in times_s]
    if any(t < 0 for t in times):
        raise ValueError("times must be non-negative")
    lim = limits
    v, assumed, capped = prepare_velocity(v0, KinematicLimits(max_speed_mps=lim.max_speed_mps))
    n = lim.n_headings
    u = _headings(n)
    lag = lim.velocity_lag_s
    start = p - v * lag
    run = sorted({t + lag for t in times if t + lag > 0})
    ends = _v2_endpoints(start, v, run, lim) if run else {}
    out = np.empty((len(times), *p.shape[:-1], n, 2), dtype=np.float64)
    base = p @ u.T  # (..., n): support of the start point
    for k, t in enumerate(times):
        margin = lim.margin_m(t)
        if t + lag > 0:
            e = ends[t + lag]  # (..., n_i, 2)
            h = np.max(np.einsum("...id,kd->...ik", e, u), axis=-2)  # (..., n_k)
        else:
            h = base.copy()
        disc = base + lim.max_speed_mps * t  # unknown velocity: every admissible path
        h = np.where(assumed[..., None], disc, h) + margin
        out[k] = support_polygon(h, n)
    return out, assumed, capped


def inside_polygon(poly: F, q: npt.ArrayLike, tol: float = _TOL) -> npt.NDArray[np.bool_]:
    """Whether ``q`` (shape ``(..., 2)``) lies in the polygon ``poly`` (``(..., n, 2)``).

    Crossing-number test plus a boundary tolerance; degenerate (point) polygons contain only
    points within ``tol`` of that point.
    """
    qa = np.asarray(q, dtype=np.float64)
    x, y = qa[..., None, 0], qa[..., None, 1]
    x1, y1 = poly[..., 0], poly[..., 1]
    x2, y2 = np.roll(x1, -1, axis=-1), np.roll(y1, -1, axis=-1)
    cond = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    crossings = np.sum(cond & (x < xint), axis=-1)
    on_boundary = boundary_distance(poly, qa) <= tol
    return np.asarray((crossings % 2 == 1) | on_boundary)


def _nearest_on_boundary(poly: F, q: F) -> tuple[F, F]:
    a = poly
    b = np.roll(poly, -1, axis=-2)
    ab = b - a
    qa = q[..., None, :]
    denom = np.sum(ab * ab, axis=-1)
    t = np.where(denom > 1e-18, np.sum((qa - a) * ab, axis=-1) / np.maximum(denom, 1e-18), 0.0)
    t = np.clip(t, 0.0, 1.0)
    foot = a + t[..., None] * ab
    d = np.linalg.norm(foot - qa, axis=-1)
    k = np.argmin(d, axis=-1)
    nearest = np.take_along_axis(foot, k[..., None, None], axis=-2)[..., 0, :]
    return nearest, np.take_along_axis(d, k[..., None], axis=-1)[..., 0]


def boundary_distance(poly: F, q: npt.ArrayLike) -> F:
    return _nearest_on_boundary(poly, np.asarray(q, dtype=np.float64))[1]


def project_onto_polygon(poly: F, q: npt.ArrayLike) -> F:
    """Closest point of the (filled) polygon to ``q``. Points inside are returned unchanged.

    For a convex polygon this is the Euclidean projection, which is non-expansive: moving
    ``q`` by ``e`` moves the result by at most ``e``.
    """
    qa = np.asarray(q, dtype=np.float64)
    nearest, _ = _nearest_on_boundary(poly, qa)
    inside = inside_polygon(poly, qa)
    return np.where(inside[..., None], qa, nearest)


@dataclass(frozen=True)
class Envelope:
    """Reachable region of one player after ``horizon_s`` seconds."""

    origin: tuple[float, float]
    velocity: tuple[float, float]  # the velocity actually used (after fallback / capping)
    horizon_s: float
    limits: Limits
    vertices: F  # (n_headings, 2)
    velocity_assumed: bool
    notes: list[str] = field(default_factory=list)

    @property
    def coast_point(self) -> tuple[float, float]:
        """Where the player would be with no acceleration (always reachable)."""
        return (
            self.origin[0] + self.velocity[0] * self.horizon_s,
            self.origin[1] + self.velocity[1] * self.horizon_s,
        )

    def contains(self, q: npt.ArrayLike) -> bool:
        return bool(inside_polygon(self.vertices, q))

    def clip(self, q: npt.ArrayLike) -> F:
        """The reachable point closest to ``q`` (``q`` itself if reachable)."""
        return np.asarray(project_onto_polygon(self.vertices, q), dtype=np.float64)


def reachability_envelope(
    p0: npt.ArrayLike,
    v0: npt.ArrayLike,
    horizon_s: float,
    limits: Limits = DEFAULT_LIMITS,
) -> Envelope:
    """Envelope for one player. ``p0`` must be finite.

    A NaN velocity falls back to a start from rest under v1, and to the speed-limit disc
    under v2.
    """
    p = np.asarray(p0, dtype=np.float64)
    if p.shape != (2,) or np.isnan(p).any():
        raise ValueError("p0 must be a finite (x, y) position")
    if isinstance(limits, KinematicLimitsV2):
        return _reachability_envelope_v2(p, np.asarray(v0, dtype=np.float64), horizon_s, limits)
    v, assumed, capped = prepare_velocity(np.asarray(v0, dtype=np.float64).reshape(2), limits)
    notes = []
    if bool(assumed):
        notes.append(
            "velocity unavailable: envelope assumes a start from rest (understates reach along "
            "an unseen run, overstates it against one)"
        )
    if bool(capped):
        notes.append(
            f"observed speed above {limits.max_speed_mps:.1f} m/s was capped to the speed limit"
        )
    verts = envelope_vertices(p, v, [horizon_s], limits)[0]
    return Envelope(
        origin=(float(p[0]), float(p[1])),
        velocity=(float(v[0]), float(v[1])),
        horizon_s=float(horizon_s),
        limits=limits,
        vertices=verts,
        velocity_assumed=bool(assumed),
        notes=notes,
    )


def _reachability_envelope_v2(p: F, v0: F, horizon_s: float, limits: KinematicLimitsV2) -> Envelope:
    verts, assumed, capped = envelope_polygons(p, v0.reshape(2), [horizon_s], limits)
    v, _, _ = prepare_velocity(v0.reshape(2), KinematicLimits(max_speed_mps=limits.max_speed_mps))
    notes = []
    if bool(assumed):
        notes.append(
            "velocity unavailable: envelope is every position reachable at "
            f"<= {limits.max_speed_mps:.1f} m/s in any direction"
        )
    if bool(capped):
        notes.append(
            f"observed speed above {limits.max_speed_mps:.1f} m/s was capped to the speed limit"
        )
    return Envelope(
        origin=(float(p[0]), float(p[1])),
        velocity=(float(v[0]), float(v[1])),
        horizon_s=float(horizon_s),
        limits=limits,
        vertices=verts[0],
        velocity_assumed=bool(assumed),
        notes=notes,
    )
