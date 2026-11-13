"""Data-quality summaries and validation gates for ingested matches."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from footymind.tracking.kinematics import (
    MAX_PLAUSIBLE_SPEED,
    implausible_motion_mask,
    speed,
    velocity,
)
from footymind.tracking.model import TrackingData

# Ingestion fails if the median passer-to-ball distance at pass start frames
# exceeds this value. The threshold detects broken event/tracking
# synchronisation, coordinate flips or player-id mismatches.
MAX_MEDIAN_PASSER_BALL_DIST_M = 1.5


class DataQualityError(RuntimeError):
    pass


def _r(x: float | np.floating | None, nd: int = 4) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def tracking_quality(td: TrackingData) -> dict[str, Any]:
    v = velocity(td.xy, td.fps, "causal", 0.2, td.period)
    sp = speed(v)
    bad = implausible_motion_mask(td.xy, td.fps, period=td.period)
    players = []
    for i, p in enumerate(td.players):
        obs = ~np.isnan(td.xy[:, i, 0])
        rows = np.flatnonzero(obs)
        s = sp[:, i]
        players.append(
            {
                "player_id": p.player_id,
                "team": p.team,
                "jersey": p.jersey,
                "observed_fraction": _r(obs.mean()),
                "first_frame": int(td.frame[rows[0]]) if rows.size else None,
                "last_frame": int(td.frame[rows[-1]]) if rows.size else None,
                "max_speed_mps": _r(np.nanmax(s)) if np.isfinite(s).any() else None,
                "frames_over_speed_limit": int(np.nansum(s > MAX_PLAUSIBLE_SPEED)),
                "implausible_step_frames": int(bad[:, i].sum()),
            }
        )
    ball_obs = ~np.isnan(td.ball_xy[:, 0])
    bx, by = td.ball_xy[ball_obs, 0], td.ball_xy[ball_obs, 1]
    outside = (np.abs(bx) > td.pitch.half_length) | (np.abs(by) > td.pitch.half_width)
    periods = {}
    for per in np.unique(td.period):
        m = td.period == per
        periods[str(int(per))] = {
            "n_frames": int(m.sum()),
            "first_frame": int(td.frame[m][0]),
            "last_frame": int(td.frame[m][-1]),
            "duration_s": _r(td.time_s[m][-1] - td.time_s[m][0] + td.dt, 2),
            "home_attacks_positive_x": td.home_attacks_positive_x[int(per)],
        }
    return {
        "n_frames": td.n_frames,
        "fps": td.fps,
        "n_players": len(td.players),
        "periods": periods,
        "ball_observed_fraction": _r(ball_obs.mean()),
        "ball_outside_pitch_fraction_of_observed": _r(outside.mean() if outside.size else 0.0),
        "implausible_step_player_frame_fraction": _r(bad.mean(), 6),
        "players": players,
    }


def event_sync_quality(td: TrackingData, events: pl.DataFrame) -> dict[str, Any]:
    """Distances between event-implied and tracked ball/player positions."""
    passes = events.filter((pl.col("type") == "PASS") & pl.col("from_player_id").is_not_null())
    idx = np.array([td.index_of_frame(f) for f in passes["start_frame"]], dtype=np.int64)
    pidx = np.array([td.player_index(x) for x in passes["from_player_id"]], dtype=np.int64)
    ball = td.ball_xy[idx]
    passer = td.xy[idx, pidx]
    d_passer = np.linalg.norm(ball - passer, axis=1)
    ev_xy = np.stack([passes["start_x"].to_numpy(), passes["start_y"].to_numpy()], axis=1)
    d_event = np.linalg.norm(ball - ev_xy, axis=1)
    with_end = passes.filter(
        pl.col("end_frame").is_not_null() & pl.col("to_player_id").is_not_null()
    )
    eidx = np.array([td.index_of_frame(f) for f in with_end["end_frame"]], dtype=np.int64)
    ridx = np.array([td.player_index(x) for x in with_end["to_player_id"]], dtype=np.int64)
    d_recv = np.linalg.norm(td.ball_xy[eidx] - td.xy[eidx, ridx], axis=1)

    def stats(a: np.ndarray) -> dict[str, float | int | None]:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"n": 0, "median": None, "p90": None}
        return {"n": int(a.size), "median": _r(np.median(a)), "p90": _r(np.percentile(a, 90))}

    return {
        "passer_to_ball_at_start_m": stats(d_passer),
        "event_start_to_ball_at_start_m": stats(d_event),
        "receiver_to_ball_at_end_m": stats(d_recv),
    }


def validate(td: TrackingData, events: pl.DataFrame) -> dict[str, Any]:
    tq = tracking_quality(td)
    sq = event_sync_quality(td, events)
    med = sq["passer_to_ball_at_start_m"]["median"]
    if med is None or med > MAX_MEDIAN_PASSER_BALL_DIST_M:
        raise DataQualityError(
            f"{td.match_id}: median passer-ball distance {med} m exceeds "
            f"{MAX_MEDIAN_PASSER_BALL_DIST_M} m; event/tracking sync is broken"
        )
    unknown = set(events["from_player_id"].drop_nulls().to_list()) - {
        p.player_id for p in td.players
    }
    if unknown:
        raise DataQualityError(f"{td.match_id}: events reference unknown players {sorted(unknown)}")
    frames = set(td.frame.tolist())
    missing = [f for f in events["start_frame"].to_list() if f not in frames]
    if missing:
        raise DataQualityError(f"{td.match_id}: {len(missing)} events reference missing frames")
    return {
        "tracking": tq,
        "event_sync": sq,
        "events_by_type": dict(events.group_by("type").len().sort("type").iter_rows()),
    }
