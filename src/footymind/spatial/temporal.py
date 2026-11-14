"""Causal temporal context at a decision frame.

Everything here is computed from frames *up to and including* the decision frame. The
public helpers take array histories that end at the decision frame, so they cannot see the
future by construction, and :func:`temporal_context` slices the tracking arrays that way.
A regression test corrupts every later frame and checks that nothing changes.

Pre-specified design (session 3, written before any result was seen):

* ``passer_control_s``: how long the ball has been within ``CONTROL_RADIUS_M`` of the
  passer, counted backwards from the decision frame and capped at ``MAX_CONTROL_S``. This is
  time-on-ball from tracking. The run stops at the first frame where the ball is unobserved,
  so it is a lower bound when the ball is lost.
* ``quick_release``: ``passer_control_s < QUICK_RELEASE_S`` (a one- or two-touch pass).
* Movement over a lookback of ``lookback_s`` (primary 1.0 s; 0.5 s and 2.0 s are sensitivity
  runs only). These are pressure accumulation, ball travel, forward movement of the passer
  and of both team centroids, change in defensive shape, and change in the number of
  teammates with an open straight lane from the passer.

Units: metres and seconds, in the attacking frame (the attacking team attacks +x). All
values are DERIVED. NaN means the history is not observed (start of a period, a missing
player or an unobserved ball).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from footymind.spatial.lanes import point_segment_distance
from footymind.tracking.model import TrackingData

F = npt.NDArray[np.float64]

TEMPORAL_LOOKBACK_S = 1.0  # primary, pre-specified
LOOKBACK_SENSITIVITY_S = (0.5, 2.0)
CONTROL_RADIUS_M = 2.0
MAX_CONTROL_S = 10.0
QUICK_RELEASE_S = 1.0
OPEN_LANE_PERP_M = 3.0  # a lane is "open" if no defender is within this distance of it

TRACK_FEATURE_STEMS = [
    "press_nearest_def_change_m",
    "press_def_within_5m_change",
    "ball_travel_m",
    "passer_fwd_m",
    "att_centroid_fwd_m",
    "def_centroid_fwd_m",
    "def_compactness_change_m",
    "def_line_depth_change_m",
    "open_lanes_change",
]


def lookback_suffix(lookback_s: float) -> str:
    """Column suffix for a lookback: 1.0 s -> ``_l10``, 0.5 s -> ``_l05``."""
    return f"_l{round(lookback_s * 10):02d}"


def track_feature_names(lookback_s: float = TEMPORAL_LOOKBACK_S) -> list[str]:
    return [s + lookback_suffix(lookback_s) for s in TRACK_FEATURE_STEMS]


def control_duration_s(passer_xy: F, ball_xy: F, fps: float) -> float:
    """Time the ball has been within ``CONTROL_RADIUS_M`` of the passer.

    ``passer_xy`` and ``ball_xy`` are (T, 2) histories that END at the decision frame.
    The count stops at the first frame (going back) where the ball is out of reach. It is
    NaN (unknown, never a fabricated number) if the passer or the ball is unobserved at the
    decision frame itself, or if the run is cut short by an *unobserved* frame (censored:
    for example the dead-ball spell before a restart, independent review 2026-11-14).
    """
    if np.isnan(passer_xy[-1]).any() or np.isnan(ball_xy[-1]).any():
        return np.nan
    d = np.linalg.norm(ball_xy - passer_xy, axis=1)
    observed = np.isfinite(d)
    ok = observed & (d <= CONTROL_RADIUS_M)
    if not ok[-1]:
        return 0.0
    if ok.all():
        return float(min(ok.size / fps, MAX_CONTROL_S))
    run = int(np.argmin(ok[::-1]))
    if run / fps >= MAX_CONTROL_S:
        return MAX_CONTROL_S
    stop = ok.size - 1 - run  # index of the frame that ended the run
    return np.nan if not observed[stop] else float(run / fps)


def _open_lanes(passer: F, mates: F, defenders: F) -> float:
    if np.isnan(passer).any():
        return np.nan
    m = mates[~np.isnan(mates).any(axis=1)]
    dfn = defenders[~np.isnan(defenders).any(axis=1)]
    if m.size == 0:
        return 0.0
    if dfn.size == 0:
        return float(len(m))
    n = 0
    for t in m:
        n += int(np.min(point_segment_distance(dfn, passer, t)) > OPEN_LANE_PERP_M)
    return float(n)


def movement_features(
    xy_then: F,
    xy_now: F,
    ball_path: F,
    att: npt.NDArray[np.bool_],
    outfield_def: npt.NDArray[np.bool_],
    passer: int,
    sign: int,
    lookback_s: float,
) -> dict[str, float]:
    """Movement between ``lookback_s`` ago (``*_then``) and the decision frame (``*_now``).

    ``ball_path`` is the (k+1, 2) ball history from then to now (inclusive).
    ``att``/``outfield_def`` are player masks. ``sign`` maps to the attacking frame.
    """
    sfx = lookback_suffix(lookback_s)
    out = {s + sfx: np.nan for s in TRACK_FEATURE_STEMS}
    p_now, p_then = xy_now[passer], xy_then[passer]
    def_idx = np.flatnonzero(outfield_def)
    # pressure accumulation: the defender nearest the passer NOW, tracked back in time
    if not np.isnan(p_now).any() and def_idx.size:
        d_now = np.linalg.norm(xy_now[def_idx] - p_now, axis=1)
        if np.isfinite(d_now).any():
            j = def_idx[int(np.nanargmin(d_now))]
            if not (np.isnan(p_then).any() or np.isnan(xy_then[j]).any()):
                out["press_nearest_def_change_m" + sfx] = float(
                    np.nanmin(d_now) - np.linalg.norm(xy_then[j] - p_then)
                )
        if not np.isnan(p_then).any():
            d_then = np.linalg.norm(xy_then[def_idx] - p_then, axis=1)
            both = np.isfinite(d_now) & np.isfinite(d_then)
            if both.any():
                out["press_def_within_5m_change" + sfx] = float(
                    (d_now[both] < 5).sum() - (d_then[both] < 5).sum()
                )
    if not np.isnan(ball_path).any():
        out["ball_travel_m" + sfx] = float(np.linalg.norm(np.diff(ball_path, axis=0), axis=1).sum())
    if not (np.isnan(p_now).any() or np.isnan(p_then).any()):
        out["passer_fwd_m" + sfx] = float((p_now[0] - p_then[0]) * sign)

    def centroid_fwd(mask: npt.NDArray[np.bool_]) -> float:
        both = mask & ~np.isnan(xy_now).any(axis=1) & ~np.isnan(xy_then).any(axis=1)
        if not both.any():
            return np.nan
        return float((xy_now[both, 0].mean() - xy_then[both, 0].mean()) * sign)

    out["att_centroid_fwd_m" + sfx] = centroid_fwd(att)
    out["def_centroid_fwd_m" + sfx] = centroid_fwd(outfield_def)
    both = outfield_def & ~np.isnan(xy_now).any(axis=1) & ~np.isnan(xy_then).any(axis=1)
    if both.sum() >= 2:
        now, then = xy_now[both], xy_then[both]
        out["def_compactness_change_m" + sfx] = float(
            np.mean(np.std(now, axis=0)) - np.mean(np.std(then, axis=0))
        )
        out["def_line_depth_change_m" + sfx] = float(np.ptp(now[:, 0]) - np.ptp(then[:, 0]))
    mates = att.copy()
    mates[passer] = False
    defenders = ~att & ~np.isnan(xy_now).any(axis=1)
    lanes_now = _open_lanes(p_now, xy_now[mates], xy_now[defenders])
    # defenders on the pitch now, at their earlier positions (not off-pitch attackers)
    lanes_then = _open_lanes(p_then, xy_then[mates], xy_then[defenders])
    if np.isfinite(lanes_now) and np.isfinite(lanes_then):
        out["open_lanes_change" + sfx] = lanes_now - lanes_then
    return out


def temporal_context(
    td: TrackingData,
    i0: int,
    team: str,
    passer: int,
    is_gk: npt.NDArray[np.bool_],
    lookbacks_s: tuple[float, ...] = (TEMPORAL_LOOKBACK_S,),
) -> dict[str, float]:
    """Temporal features at tracking index ``i0`` (the decision frame) for ``passer``.

    Only frames ``<= i0`` of the same period are read.
    """
    period = int(td.period[i0])
    sign = td.attack_sign(team, period)  # type: ignore[arg-type]
    teams = np.array([p.team for p in td.players])
    att = teams == team
    on_now = ~np.isnan(td.xy[i0]).any(axis=1)
    outfield_def = (~att) & ~is_gk
    # earliest index of this period that is not after i0
    lo = i0
    max_back = round(MAX_CONTROL_S * td.fps)
    while lo > 0 and i0 - lo < max_back and td.period[lo - 1] == period:
        lo -= 1
    hist_xy = td.xy[lo : i0 + 1].astype(np.float64)
    hist_ball = td.ball_xy[lo : i0 + 1].astype(np.float64)
    out: dict[str, float] = {
        "passer_control_s": control_duration_s(hist_xy[:, passer], hist_ball, td.fps),
    }
    tob = out["passer_control_s"]
    out["quick_release"] = float(tob < QUICK_RELEASE_S) if np.isfinite(tob) else np.nan
    for lb in lookbacks_s:
        k = round(lb * td.fps)
        if i0 - k < lo:  # not enough same-period history
            out.update({s + lookback_suffix(lb): np.nan for s in TRACK_FEATURE_STEMS})
            continue
        out.update(
            movement_features(
                xy_then=hist_xy[-1 - k],
                xy_now=hist_xy[-1],
                ball_path=hist_ball[-1 - k :],
                att=att & on_now,
                outfield_def=outfield_def & on_now,
                passer=passer,
                sign=sign,
                lookback_s=lb,
            )
        )
    return out


def history_detected_fraction(
    td: TrackingData, i0: int, lookback_s: float = TEMPORAL_LOOKBACK_S
) -> float:
    """Share of on-pitch player positions in the lookback window that are detected.

    Used to stratify results by tracking quality. It is never a model feature. It is 1.0
    for optical tracking (Metrica), where every drawn position is observed.
    """
    if td.detected is None:
        return 1.0
    k = round(lookback_s * td.fps)
    lo = max(0, i0 - k)
    while td.period[lo] != td.period[i0]:  # never cross a period boundary
        lo += 1
    on = ~np.isnan(td.xy[lo : i0 + 1]).any(axis=2)
    det = td.detected[lo : i0 + 1] & on
    return float(int(det.sum()) / max(int(on.sum()), 1))
