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

    def describe(self) -> str:
        return (
            f"{self.reaction_s:.2f} s reaction, <= {self.max_accel_mps2:.1f} m/s^2, "
            f"<= {self.max_speed_mps:.1f} m/s"
        )


DEFAULT_LIMITS = KinematicLimits()


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
    limits: KinematicLimits
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
    limits: KinematicLimits = DEFAULT_LIMITS,
) -> Envelope:
    """Envelope for one player. ``p0`` must be finite; a NaN velocity falls back to rest."""
    p = np.asarray(p0, dtype=np.float64)
    if p.shape != (2,) or np.isnan(p).any():
        raise ValueError("p0 must be a finite (x, y) position")
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
