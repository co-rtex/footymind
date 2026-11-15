"""Passing-lane geometry and a heuristic interception model.

Quantities and their status:

* ``perpendicular distance`` and ``time margin`` of each defender to a pass
  lane are **derived** geometric or kinematic quantities.
* ``p_intercept`` and ``p_lane_clear`` are **estimated** by a heuristic
  logistic model that shares the pitch-control parameters. They are not
  calibrated. ``footymind.research`` tests whether these quantities carry
  predictive signal for real pass outcomes.

Lane model: the ball travels in a straight line from ``ball`` to ``target`` at
constant speed ``v_ball``. For sample points ``s`` on the segment (excluding
the first metre, which belongs to the passer, and the final ``end_exclusion_m``,
where reception at the target is handled by pitch control), defender ``d``
reaches ``s`` at ``tau_d(s)`` and the ball at ``t_b(s)``. The time margin is
``m_d = min_s (tau_d(s) - t_b(s))``. A negative margin means the defender can
reach the lane before the ball passes. The heuristic interception probability
is ``p_d = sigmoid(-m_d * pi / (sqrt(3) sigma))``, and lane clearance is
``prod_d (1 - p_d)``. Defenders are treated as independent, which is a stated
simplification.

Players are not points: a defender can block the ball within ``reach_m``
(default 1 m, about a stride or leg reach) of their position, so the distance to
cover is reduced by ``reach_m``.

Known limitation: the reaction time applies uniformly, as in the PPCF
model. A defender standing almost in the lane of a *short, fast* pass is
therefore given a moderate, not high, interception probability. The
research experiments measure whether this matters empirically. The
derived perpendicular-distance features do not share this limitation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from footymind.spatial.pitch_control import DEFAULT_PARAMS, PitchControlParams

F = npt.NDArray[np.float64]


@dataclass(frozen=True)
class LaneResult:
    length_m: float
    ball_travel_time_s: float
    min_margin_s: float  # nan if no eligible defender
    closest_defender_index: int | None  # index into def arrays
    closest_defender_perp_m: float
    n_defenders_within_3m_of_lane: int
    p_lane_clear: float
    p_intercept_by_defender: F


def point_segment_distance(p: F, a: F, b: F) -> F:
    """Distance from points ``p`` (N,2) to the segment ``a``-``b``."""
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return np.linalg.norm(p - a, axis=1)
    t = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(p - proj, axis=1)


def lane(
    ball: F,
    target: F,
    def_pos: F,
    def_vel: F,
    params: PitchControlParams = DEFAULT_PARAMS,
    n_samples: int = 24,
    start_exclusion_m: float = 1.0,
    end_exclusion_m: float = 2.0,
    reach_m: float = 1.0,
) -> LaneResult:
    ball = np.asarray(ball, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    vec = target - ball
    length = float(np.linalg.norm(vec))
    t_total = length / params.average_ball_speed
    ok = ~np.isnan(def_pos[:, 0])
    perp = np.full(def_pos.shape[0], np.nan)
    perp[ok] = point_segment_distance(def_pos[ok], ball, target)
    within3 = int(np.nansum(perp < 3.0))
    p_int = np.zeros(def_pos.shape[0])
    lo, hi = start_exclusion_m, length - end_exclusion_m
    if hi <= lo or not ok.any():
        return LaneResult(length, t_total, np.nan, None, np.nan, within3, 1.0, p_int)
    s = np.linspace(lo, hi, n_samples)
    pts = ball[None, :] + (s / length)[:, None] * vec[None, :]
    t_b = s / params.average_ball_speed
    react = def_pos[ok] + np.nan_to_num(def_vel[ok]) * params.reaction_time
    dist = np.linalg.norm(pts[:, None, :] - react[None, :, :], axis=-1)  # (S, D)
    tau = params.reaction_time + np.maximum(dist - reach_m, 0.0) / params.max_player_speed
    margin = (tau - t_b[:, None]).min(axis=0)  # (D,)
    k = np.pi / (np.sqrt(3.0) * params.tti_sigma)
    p_ok = 1.0 / (1.0 + np.exp(k * margin))
    p_int[ok] = p_ok
    full_margin = np.full(def_pos.shape[0], np.inf)
    full_margin[ok] = margin
    ci = int(np.argmin(full_margin))
    return LaneResult(
        length_m=length,
        ball_travel_time_s=t_total,
        min_margin_s=float(full_margin[ci]),
        closest_defender_index=ci,
        closest_defender_perp_m=float(perp[ci]),
        n_defenders_within_3m_of_lane=within3,
        p_lane_clear=float(np.prod(1.0 - p_ok)),
        p_intercept_by_defender=p_int,
    )
