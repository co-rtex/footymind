"""Leakage-safe dataset of attempted passes for outcome prediction.

Research question (operationalised): does tracking-derived spatial context
improve the prediction of whether an *attempted* pass is completed, beyond
event-level information available at the moment of the pass?

Labels (Metrica event semantics, verified on the sample files)
--------------------------------------------------------------
* success = ``PASS`` (Metrica records only completed passes as PASS, with a receiver).
* failure = ``BALL LOST`` whose subtype indicates a failed pass/kick, plus
  (optionally) ``BALL OUT`` that is not a shot or clearance. Excluded are
  dispossessions (``THEFT``, ``FORCED``), ``HAND BALL``, ``OFFSIDE``,
  ``END HALF``, ``REFEREE HIT``, ``WOODWORK`` and all clearances. These are
  not pass attempts, or their outcome is not a pass outcome.

Information cutoff
------------------
Features use only:

* tracking at the event ``start_frame`` (positions) and causal velocities
  (backward 0.2 s window), and
* the *action* itself: the ball's release direction and speed, estimated
  over at most ``RELEASE_WINDOW_S`` seconds after the start frame and
  truncated so the window never reaches the outcome frame (``end_frame``),
* event history strictly before the pass.

Explicitly excluded because they leak the label: ``End X/Y`` (for failures,
this is the loss location, not the intended target), ``end_frame`` and the pass
duration, the receiver id (only known for successes), and event subtypes that
encode the outcome (e.g. ``INTERCEPTION``).

Intended target. For failures the intended target is unobserved. The
spatial models infer a *candidate* receiver from the release direction: the
teammate minimising angular deviation plus a distance penalty. The inference is
validated against the true receiver on completed passes (see
``inferred_receiver_accuracy``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from footymind.data.store import load_match
from footymind.spatial.lanes import lane, point_segment_distance
from footymind.spatial.option_features import option_feature_rows
from footymind.spatial.pitch_control import DEFAULT_PARAMS, pitch_control
from footymind.spatial.state import frame_state
from footymind.spatial.temporal import (
    LOOKBACK_SENSITIVITY_S,
    TEMPORAL_LOOKBACK_S,
    history_detected_fraction,
    temporal_context,
    track_feature_names,
)
from footymind.tracking.model import TrackingData

RELEASE_WINDOW_S = 0.12  # 3 frames at 25 Hz (Metrica), 1 frame at 10 Hz (SkillCorner)

# Label rules are token-based: a subtype such as "CROSS-INTERCEPTION" (CSV) or
# ["CROSS", "INTERCEPTION"] (JSON) qualifies if all of its tokens are allowed. A
# null subtype qualifies. Any other token (THEFT, FORCED, CLEARANCE, HAND BALL,
# OFFSIDE, END HALF, REFEREE HIT, WOODWORK, ...) excludes the event.
FAILED_PASS_LOST_TOKENS = frozenset({"INTERCEPTION", "HEAD", "CROSS", "GOAL KICK", "DEEP BALL"})
FAILED_PASS_OUT_TOKENS = frozenset({"HEAD", "CROSS", "GOAL KICK"})
# v2: one allow-list of pass-attempt subtype tokens for successes and failures alike.
PASS_TOKENS_V2 = frozenset({"HEAD", "CROSS", "GOAL KICK", "DEEP BALL", "THROUGH BALL"})
FAILED_PASS_LOST_TOKENS_V2 = PASS_TOKENS_V2 | {"INTERCEPTION"}
FAILED_PASS_OUT_TOKENS_V2 = PASS_TOKENS_V2
# Event types excluded from event-history features. CARRY exists only in the EPTS/JSON
# format (Game 3). Including it would shift "previous event" distributions between matches.
HISTORY_EXCLUDED_TYPES = frozenset({"CARRY"})

EVENT_FEATURES = [
    "ball_x_att",
    "ball_y_abs",
    "dist_to_goal",
    "angle_to_goal",
    "release_cos",
    "release_sin_abs",
    "release_speed",
    "is_head",
    "is_cross",
    "is_goal_kick",
    "is_restart",
    "prev_type_pass",
    "prev_type_recovery",
    "prev_type_challenge",
    "possession_pass_index",
    "time_in_possession_s",
    "period",
    "match_minute",
]
SPATIAL_FEATURES = [
    "press_nearest_def_m",
    "press_closing_speed",
    "press_def_within_5m",
    "press_def_within_10m",
    "cand_distance_m",
    "cand_angle_dev",
    "cand_forward_m",
    "cand_ppcf",
    "cand_p_lane_clear",
    "cand_lane_margin_s",
    "cand_lane_perp_m",
    "cand_def_within_3m_lane",
    "cand_p_success_physics",
    "ray_min_def_perp_m",
    "ray_def_in_cone",
    "passer_speed",
]
# Strictly decision-time feature sets: nothing derived from the executed kick (no release
# direction/speed, no direction-inferred candidate receiver, no release-ray features, no
# body-part/cross annotations of the pass itself). They answer "before the pass is played,
# how likely is the possessor's next pass to be completed?" (independent review, 2026-11-13).
DECISION_EVENT_FEATURES = [
    "ball_x_att",
    "ball_y_abs",
    "dist_to_goal",
    "angle_to_goal",
    "is_goal_kick",
    "is_restart",
    "prev_type_pass",
    "prev_type_recovery",
    "prev_type_challenge",
    "possession_pass_index",
    "time_in_possession_s",
    "period",
    "match_minute",
]
DECISION_SPATIAL_FEATURES = [
    *DECISION_EVENT_FEATURES,
    "press_nearest_def_m",
    "press_closing_speed",
    "press_def_within_5m",
    "press_def_within_10m",
    "passer_speed",
    "best_option_p_success_physics",
    "n_options_over_0_5",
    "def_goal_side_of_ball",
    "def_line_depth_m",
    "def_compactness_m",
    "att_width_m",
    "att_depth_m",
]

# Feature groups for ablations. "Defender-only" groups use no teammate positions.
GROUP_PRESSURE = [
    "press_nearest_def_m",
    "press_closing_speed",
    "press_def_within_5m",
    "press_def_within_10m",
    "passer_speed",
]
GROUP_RAY_DEFENSE = ["ray_min_def_perp_m", "ray_def_in_cone"]
GROUP_TARGET_GEOMETRY = ["cand_distance_m", "cand_angle_dev", "cand_forward_m"]
GROUP_TARGET_DEFENSE = [
    "cand_ppcf",
    "cand_p_lane_clear",
    "cand_lane_margin_s",
    "cand_lane_perp_m",
    "cand_def_within_3m_lane",
    "cand_p_success_physics",
]

# Temporal context (session 3, pre-specified; see spatial/temporal.py). All causal.
# Event-derived timing: available to an event-only baseline.
TEMPORAL_EVENT_FEATURES = ["time_since_prev_event_s"]
# Time-on-ball from tracking (the ball within 2 m of the passer, counted backwards).
TIME_ON_BALL_FEATURES = ["passer_control_s", "quick_release"]
# Movement over the primary 1.0 s lookback; the 0.5 s / 2.0 s variants are sensitivity runs.
TEMPORAL_TRACK_FEATURES = track_feature_names(TEMPORAL_LOOKBACK_S)
TEMPORAL_TRACK_FEATURES_L05 = track_feature_names(LOOKBACK_SENSITIVITY_S[0])
TEMPORAL_TRACK_FEATURES_L20 = track_feature_names(LOOKBACK_SENSITIVITY_S[1])

_TEMPORAL_COLS = frozenset(
    [
        *TIME_ON_BALL_FEATURES,
        *TEMPORAL_TRACK_FEATURES,
        *TEMPORAL_TRACK_FEATURES_L05,
        *TEMPORAL_TRACK_FEATURES_L20,
    ]
)

INTERACTION_FEATURES = [
    "cand2_p_success_physics",
    "cand3_p_success_physics",
    "best_option_p_success_physics",
    "n_options_over_0_5",
    "cand_marker_dist_m",
    "def_goal_side_of_ball",
    "def_line_depth_m",
    "def_compactness_m",
    "att_width_m",
    "att_depth_m",
    "sum_p_intercept_ray",
]


@dataclass(frozen=True)
class DatasetConfig:
    # Label/feature rule set. "v1" reproduces the committed v1/v2 experiments exactly.
    # "v2" (default) applies the independent-review fixes (docs/RESEARCH.md, "Review"):
    #   * PASS events and failures use the same subtype allow-list (no clearances or
    #     offsides on either side; through balls on both sides);
    #   * history features only use events in strictly earlier frames; the restart flag
    #     comes from the possession's last set piece;
    #   * attempts resolved in < 2 frames are dropped, because the release window would
    #     touch the outcome frame;
    #   * release speeds above max_release_speed are dropped as tracking artefacts.
    rules: str = "v2"
    max_release_speed: float = 40.0  # m/s, v2 only
    include_ball_out: bool = True
    min_release_speed: float = 1.0  # m/s; below this the release is not observable
    cone_half_angle_deg: float = 15.0
    # Leakage guard (session 3): compute the temporal features this many seconds BEFORE the
    # release frame, so no frame near the kick can enter them. 0 = at the release frame.
    temporal_offset_s: float = 0.0
    # Leakage guard for the *static* decision-time features (positions, pressure, structure):
    # read the frame this many seconds before the release frame. On SkillCorner (10 fps) the
    # ball is often already leaving the foot at the event's start frame (independent review,
    # 2026-11-14), so release-frame positions can carry the start of the kick. Release and
    # label features are unaffected.
    static_offset_s: float = 0.0
    # Opt-in (default off so committed results reproduce): censor time-on-ball at frames where
    # the provider extrapolated the ball (SkillCorner ``ball_detected``; needs re-ingested
    # bundles). Planned for a pre-registered SkillCorner re-run (CURRENT_STATE, next actions).
    censor_undetected_ball: bool = False
    ray_length_m: float = 40.0


def label_events(ev: pl.DataFrame, cfg: DatasetConfig) -> pl.DataFrame:
    """Select attempted passes and attach the binary label ``success``."""
    tokens = pl.col("subtype").fill_null("").str.split("-")

    def allowed(tok: frozenset[str]) -> pl.Expr:
        return (
            pl.col("subtype").is_null() | tokens.list.eval(pl.element().is_in(list(tok))).list.all()
        )

    if cfg.rules == "v1":
        is_pass = pl.col("type") == "PASS"
        is_lost = (pl.col("type") == "BALL LOST") & allowed(FAILED_PASS_LOST_TOKENS)
        is_out = (pl.col("type") == "BALL OUT") & allowed(FAILED_PASS_OUT_TOKENS)
    elif cfg.rules == "v2":
        is_pass = (pl.col("type") == "PASS") & allowed(PASS_TOKENS_V2)
        is_lost = (pl.col("type") == "BALL LOST") & allowed(FAILED_PASS_LOST_TOKENS_V2)
        is_out = (pl.col("type") == "BALL OUT") & allowed(FAILED_PASS_OUT_TOKENS_V2)
    else:
        raise ValueError(f"unknown rules {cfg.rules!r}")
    cond = is_pass | is_lost | (is_out if cfg.include_ball_out else pl.lit(False))
    return (
        ev.filter(cond & pl.col("from_player_id").is_not_null())
        .with_columns(is_pass.cast(pl.Int8).alias("success"))
        .with_columns(
            pl.col("type").alias("label_source"),
        )
    )


def _prev_event_features(ev: pl.DataFrame) -> pl.DataFrame:
    """Per event: previous event type in the same possession, pass index, time in possession."""
    # Raw previous event (including CARRY), kept as label-side metadata for the
    # symmetric "after carry" sensitivity filter. It is never a model feature.
    ev = ev.sort(["start_frame", "event_id"]).with_columns(
        pl.col("type").shift(1).alias("prev_raw_type"),
        (pl.col("from_player_id").shift(1) == pl.col("from_player_id")).alias(
            "prev_raw_same_player"
        ),
    )
    ev = ev.filter(~pl.col("type").is_in(list(HISTORY_EXCLUDED_TYPES)))
    ev = ev.with_columns(
        pl.col("type").shift(1).over("possession_id").alias("prev_type"),
        pl.col("subtype").shift(1).over("possession_id").alias("prev_subtype"),
        pl.col("start_frame").shift(1).over("possession_id").alias("prev_start_frame"),
        (pl.col("type") == "PASS")
        .cast(pl.Int32)
        .cum_sum()
        .shift(1, fill_value=0)
        .over("possession_id")
        .alias("possession_pass_index"),
        pl.col("start_time_s").min().over("possession_id").alias("possession_start_s"),
    )
    # v2 history: the last event in a strictly earlier frame of the same possession, and the
    # frame of the possession's most recent set piece (same frame included: a restart is
    # known before the pass is played).
    frames = (
        ev.group_by(["possession_id", "start_frame"], maintain_order=True)
        .agg(pl.col("type").last().alias("_last_type"))
        .sort(["possession_id", "start_frame"])
        .with_columns(
            pl.col("_last_type").shift(1).over("possession_id").alias("prev_type_strict"),
            pl.col("start_frame").shift(1).over("possession_id").alias("prev_frame_strict"),
        )
        .drop("_last_type")
    )
    ev = ev.join(frames, on=["possession_id", "start_frame"], how="left").with_columns(
        pl.when(pl.col("type") == "SET PIECE")
        .then(pl.col("start_frame"))
        .otherwise(None)
        .forward_fill()
        .over("possession_id")
        .alias("last_set_piece_frame")
    )
    return ev


def _angle_dev(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unsigned angle in radians between direction ``a`` (2,) and vectors ``b`` (N,2)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b, axis=1)
    cos = (b @ a) / np.maximum(na * nb, 1e-9)
    return np.arccos(np.clip(cos, -1, 1))


def build(match_ids: list[str], cfg: DatasetConfig = DatasetConfig()) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for mid in match_ids:
        b = load_match(mid)
        td = b.tracking
        ev = _prev_event_features(b.events)
        cand = label_events(ev, cfg)
        for r in cand.iter_rows(named=True):
            row = _features_for(mid, r, td, cfg)
            if row is not None:
                rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def _features_for(
    mid: str, r: dict[str, Any], td: TrackingData, cfg: DatasetConfig
) -> dict[str, object] | None:
    f0 = int(r["start_frame"])
    i0 = td.index_of_frame(f0)
    # At low frame rates one frame often precedes the ball leaving the foot (SkillCorner review),
    # so at least 2 frames are used below 20 fps. Metrica (25 Hz) keeps 3 frames.
    k_full = max(2 if td.fps < 20 else 1, round(RELEASE_WINDOW_S * td.fps))
    k = k_full
    if cfg.rules == "v2" and r["end_frame"] is not None and int(r["end_frame"]) - f0 < 2:
        return None  # resolved before a pre-outcome release window exists
    if r["end_frame"] is not None:
        k = min(k, max(1, int(r["end_frame"]) - f0 - 1))
    if i0 + k >= td.n_frames or td.period[i0 + k] != td.period[i0]:
        return None
    b0, b1 = td.ball_xy[i0].astype(float), td.ball_xy[i0 + k].astype(float)
    if np.isnan(b0).any() or np.isnan(b1).any():
        return None
    rel = (b1 - b0) / (k / td.fps)
    speed = float(np.linalg.norm(rel))
    if speed < cfg.min_release_speed:
        return None
    if cfg.rules == "v2" and speed > cfg.max_release_speed:
        return None

    i_s = i0 - round(cfg.static_offset_s * td.fps)
    if i_s < 0 or td.period[i_s] != td.period[i0]:
        return None
    s = frame_state(mid, int(td.frame[i_s]))
    team = r["team"]
    sign = td.attack_sign(team, int(r["period"]))
    passer = td.player_index(r["from_player_id"])
    ball = s.ball if not np.isnan(s.ball).any() else b0
    # attacking frame (team attacks +x)
    bx, by = ball * sign
    d_dir = rel * sign  # release direction in attacking frame
    gx = td.pitch.half_length
    goal = np.array([gx, 0.0])
    to_goal = goal - np.array([bx, by])
    dist_goal = float(np.linalg.norm(to_goal))
    # opening angle of the goal mouth (7.32 m) from the ball
    a1 = np.arctan2(3.66 - by, gx - bx)
    a2 = np.arctan2(-3.66 - by, gx - bx)
    st = (r["subtype"] or "").upper()
    restart_v2 = False
    if cfg.rules == "v2":
        prev = r["prev_type_strict"]
        lsp = r["last_set_piece_frame"]
        restart_v2 = lsp is not None and 0 <= f0 - int(lsp) <= 2
    else:
        prev = r["prev_type"]
    restart = (
        restart_v2
        if cfg.rules == "v2"
        else (
            prev == "SET PIECE"
            and r["prev_start_frame"] is not None
            and (f0 - int(r["prev_start_frame"]) <= 2)
        )
    )
    row: dict[str, object] = {
        "match_id": mid,
        "event_id": int(r["event_id"]),
        "possession_id": r["possession_id"],
        "frame": f0,
        "team": team,
        "passer_id": r["from_player_id"],
        "receiver_id": r["to_player_id"],  # label-side info: never a feature
        "success": int(r["success"]),
        "label_source": r["label_source"],
        "release_window_frames": k,
        "release_truncated": int(k < k_full),
        "after_own_carry": int(r["prev_raw_type"] == "CARRY" and bool(r["prev_raw_same_player"])),
        "ball_x_att": float(bx),
        "ball_y_abs": float(abs(by)),
        "dist_to_goal": dist_goal,
        "angle_to_goal": float(abs(a1 - a2)),
        "release_cos": float(d_dir[0] / speed),
        "release_sin_abs": float(abs(d_dir[1]) / speed),
        "release_speed": speed,
        "is_head": int("HEAD" in st),
        "is_cross": int("CROSS" in st),
        "is_goal_kick": int("GOAL KICK" in st),
        "is_restart": int(bool(restart)),
        "prev_type_pass": int(prev == "PASS"),
        "prev_type_recovery": int(prev == "RECOVERY"),
        "prev_type_challenge": int(prev == "CHALLENGE"),
        "possession_pass_index": int(r["possession_pass_index"] or 0),
        "time_in_possession_s": float(r["start_time_s"] - r["possession_start_s"]),
        "period": int(r["period"]),
        "match_minute": float(r["start_time_s"] / 60.0),
        # strictly earlier event of the same possession (v2 history); NaN if none
        "time_since_prev_event_s": (
            float((f0 - int(r["prev_frame_strict"])) / td.fps)
            if r.get("prev_frame_strict") is not None
            else np.nan
        ),
    }

    i_t = i0 - round(cfg.temporal_offset_s * td.fps)
    lookbacks = (TEMPORAL_LOOKBACK_S, *LOOKBACK_SENSITIVITY_S)
    if i_t < 0 or td.period[i_t] != td.period[i0]:
        row.update(
            temporal_context(td, i0, team, passer, s.is_gk, lookbacks, cfg.censor_undetected_ball)
        )
        row.update({k: np.nan for k in row if k in _TEMPORAL_COLS})
    else:
        row.update(
            temporal_context(td, i_t, team, passer, s.is_gk, lookbacks, cfg.censor_undetected_ball)
        )
    # tracking-quality metadata for stratified analysis (never a feature)
    row["history_detected_frac"] = history_detected_fraction(td, max(i_t, 0))

    att = s.team_mask(team) & s.on_pitch()
    dfn = (~s.team_mask(team)) & s.on_pitch()
    att_idx = np.flatnonzero(att)
    def_idx = np.flatnonzero(dfn)
    dxy, dvel = s.xy[def_idx], s.vel[def_idx]
    if def_idx.size == 0:
        return None  # no defender positions to describe (only seen at guarded offset frames)

    # pressure on the passer (DERIVED)
    rel_d = dxy - s.xy[passer][None, :]
    dd = np.linalg.norm(rel_d, axis=1)
    j = int(np.argmin(dd))
    closing = np.nan
    if not (np.isnan(dvel[j]).any() or np.isnan(s.vel[passer]).any()) and dd[j] > 0:
        closing = float(-(dvel[j] - s.vel[passer]) @ (rel_d[j] / dd[j]))
    row.update(
        press_nearest_def_m=float(dd[j]),
        press_closing_speed=closing,
        press_def_within_5m=int((dd < 5).sum()),
        press_def_within_10m=int((dd < 10).sum()),
        passer_speed=float(np.linalg.norm(s.vel[passer]))
        if not np.isnan(s.vel[passer]).any()
        else np.nan,
    )

    # candidate receivers ranked by alignment with the release direction
    mates = np.array([i for i in att_idx if i != passer])
    vecs = s.xy[mates] - ball[None, :]
    dist = np.linalg.norm(vecs, axis=1)
    dev = _angle_dev(rel, vecs)
    score = dev + 0.004 * dist  # prefer aligned, then nearer (0.004 rad/m)
    order = np.argsort(score)
    targets = s.xy[mates[order[:3]]]
    att_xy, att_v = s.xy[att_idx], s.vel[att_idx]
    ppcf, _, _ = pitch_control(
        targets, ball, att_xy, att_v, dxy, dvel, s.is_gk[att_idx], s.is_gk[def_idx], DEFAULT_PARAMS
    )
    phys = []
    lanes = []
    for t_i in range(targets.shape[0]):
        lr = lane(ball, targets[t_i], dxy, dvel, DEFAULT_PARAMS)
        lanes.append(lr)
        phys.append(float(ppcf[t_i]) * lr.p_lane_clear)
    c = order[0]
    lr0 = lanes[0]
    row.update(
        inferred_receiver_id=s.player_ids[mates[c]],
        target_x_att=float(s.xy[mates[c], 0] * sign),
        target_y_abs=float(abs(s.xy[mates[c], 1] * sign)),
        **{
            k: v
            for k, v in option_feature_rows(s, team, ball, [int(mates[c])], passer)[0].items()
            if k
            in (
                "target_marker_dist_m",
                "pass_forward_cos",
                "target_speed_along_pass",
                "lead_ppcf",
                "lead_p_lane_clear",
                "lead_marker_dist_m",
            )
        },
        cand_distance_m=float(dist[c]),
        cand_angle_dev=float(dev[c]),
        cand_forward_m=float((s.xy[mates[c], 0] - ball[0]) * sign),
        cand_ppcf=float(ppcf[0]),
        cand_p_lane_clear=lr0.p_lane_clear,
        cand_lane_margin_s=lr0.min_margin_s,
        cand_lane_perp_m=lr0.closest_defender_perp_m,
        cand_def_within_3m_lane=lr0.n_defenders_within_3m_of_lane,
        cand_p_success_physics=phys[0],
        cand2_p_success_physics=phys[1] if len(phys) > 1 else np.nan,
        cand3_p_success_physics=phys[2] if len(phys) > 2 else np.nan,
    )

    # ray along the release direction (independent of receiver inference)
    u = rel / speed
    end = ball + u * cfg.ray_length_m
    perp = point_segment_distance(dxy, ball, end)
    ahead = (dxy - ball[None, :]) @ u
    ang = _angle_dev(rel, dxy - ball[None, :])
    in_cone = (ang < np.deg2rad(cfg.cone_half_angle_deg)) & (ahead > 0) & (ahead < cfg.ray_length_m)
    ray_lane = lane(ball, end, dxy, dvel, DEFAULT_PARAMS)
    row.update(
        # censored at the ray length when no defender is ahead of the ball
        ray_min_def_perp_m=float(min(np.min(np.where(ahead > 0, perp, np.inf)), cfg.ray_length_m)),
        ray_def_in_cone=int(in_cone.sum()),
        sum_p_intercept_ray=float(ray_lane.p_intercept_by_defender.sum()),
    )

    # interaction / structure features
    all_targets = s.xy[mates]
    ppcf_all, _, _ = pitch_control(
        all_targets, ball, att_xy, att_v, dxy, dvel, s.is_gk[att_idx], s.is_gk[def_idx]
    )
    all_phys = [
        float(ppcf_all[m]) * lane(ball, all_targets[m], dxy, dvel).p_lane_clear
        for m in range(len(mates))
    ]
    cand_xy = s.xy[mates[c]]
    def_att = dxy * sign
    outfield = ~s.is_gk[def_idx]
    row.update(
        best_option_p_success_physics=float(max(all_phys)),
        n_options_over_0_5=int(sum(p > 0.5 for p in all_phys)),
        cand_marker_dist_m=float(np.min(np.linalg.norm(dxy - cand_xy[None, :], axis=1))),
        def_goal_side_of_ball=int((def_att[:, 0] > bx).sum()),
        def_line_depth_m=float(np.max(def_att[outfield, 0]) - np.min(def_att[outfield, 0])),
        def_compactness_m=float(np.mean(np.std(dxy[outfield], axis=0))),
        att_width_m=float(np.ptp(att_xy[:, 1])),
        att_depth_m=float(np.ptp(att_xy[:, 0])),
    )
    return row


def inferred_receiver_accuracy(df: pl.DataFrame) -> dict[str, float]:
    ok = df.filter(pl.col("success") == 1)
    acc = (ok["inferred_receiver_id"] == ok["receiver_id"]).cast(pl.Float64).mean()
    return {"n_completed": ok.height, "accuracy": float(acc or 0.0)}  # type: ignore[arg-type]
