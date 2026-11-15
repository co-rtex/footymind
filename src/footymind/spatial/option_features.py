"""Per-option features shared by training (research) and serving (analysis/scenarios).

One implementation is used for both, so train/serve skew is impossible by construction.
``tests/test_option_model.py`` checks this equality on real frames.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from footymind.spatial.lanes import lane
from footymind.spatial.pitch_control import DEFAULT_PARAMS, PitchControlParams, pitch_control
from footymind.spatial.state import FrameState

F = npt.NDArray[np.float64]
GOAL_X = 52.5


def option_feature_rows(
    state: FrameState,
    team: str,
    ball: F,
    target_indices: list[int],
    passer: int | None,
    params: PitchControlParams = DEFAULT_PARAMS,
) -> list[dict[str, float]]:
    """Features for passing from ``ball`` to each player in ``target_indices``.

    ``team`` is the passing team. ``passer`` (player index or None) defines
    pressure. The attacking direction comes from ``state`` for ``team``.
    """
    sign = state.attack_sign if state.attacking_team == team else -state.attack_sign
    att = state.team_mask(team) & state.on_pitch()  # type: ignore[arg-type]
    dfn = (~state.team_mask(team)) & state.on_pitch()  # type: ignore[arg-type]
    att_idx, def_idx = np.flatnonzero(att), np.flatnonzero(dfn)
    dxy, dvel = state.xy[def_idx], state.vel[def_idx]
    targets = state.xy[target_indices]
    ppcf, _, _ = pitch_control(
        targets,
        ball,
        state.xy[att_idx],
        state.vel[att_idx],
        dxy,
        dvel,
        state.is_gk[att_idx],
        state.is_gk[def_idx],
        params,
    )
    bx, by = ball[0] * sign, ball[1] * sign
    base = {
        "ball_x_att": float(bx),
        "ball_y_abs": float(abs(by)),
        "dist_to_goal": float(np.hypot(GOAL_X - bx, by)),
    }
    press: dict[str, float] = {
        "press_nearest_def_m": np.nan,
        "press_def_within_5m": np.nan,
        "press_def_within_10m": np.nan,
    }
    if passer is not None and def_idx.size:
        dd = np.linalg.norm(dxy - state.xy[passer][None, :], axis=1)
        press = {
            "press_nearest_def_m": float(dd.min()),
            "press_def_within_5m": float((dd < 5).sum()),
            "press_def_within_10m": float((dd < 10).sum()),
        }
    # Lead-pass targets: where the receiver will be when the ball arrives if they keep their
    # (causal) velocity. Computed with the same physics, clipped to the pitch. Unknown
    # velocity means the current position is used.
    lead_targets = np.empty_like(targets)
    for k, ti in enumerate(target_indices):
        v = np.nan_to_num(state.vel[ti])
        t_ball = float(np.linalg.norm(targets[k] - ball)) / params.average_ball_speed
        lead_targets[k] = np.clip(targets[k] + v * t_ball, [-52.5, -34.0], [52.5, 34.0])
    ppcf_lead, _, _ = pitch_control(
        lead_targets,
        ball,
        state.xy[att_idx],
        state.vel[att_idx],
        dxy,
        dvel,
        state.is_gk[att_idx],
        state.is_gk[def_idx],
        params,
    )
    rows = []
    for k, ti in enumerate(target_indices):
        t = state.xy[ti]
        lr = lane(ball, t, dxy, dvel, params)
        dist = float(np.linalg.norm(t - ball))
        unit = (t - ball) / dist if dist > 0 else np.zeros(2)
        tv = state.vel[ti]
        rows.append(
            {
                **base,
                **press,
                "target_x_att": float(t[0] * sign),
                "target_y_abs": float(abs(t[1] * sign)),
                "cand_distance_m": float(np.linalg.norm(t - ball)),
                "cand_forward_m": float((t[0] - ball[0]) * sign),
                "cand_ppcf": float(ppcf[k]),
                "cand_p_lane_clear": lr.p_lane_clear,
                "cand_lane_margin_s": lr.min_margin_s,
                "cand_lane_perp_m": lr.closest_defender_perp_m,
                "cand_def_within_3m_lane": float(lr.n_defenders_within_3m_of_lane),
                "target_marker_dist_m": float(np.linalg.norm(dxy - t[None, :], axis=1).min())
                if def_idx.size
                else np.nan,
                "pass_forward_cos": float((t[0] - ball[0]) * sign / dist) if dist > 0 else 0.0,
                "target_speed_along_pass": float(tv @ unit) if not np.isnan(tv).any() else np.nan,
                "lead_ppcf": float(ppcf_lead[k]),
                "lead_p_lane_clear": lane(ball, lead_targets[k], dxy, dvel, params).p_lane_clear,
                "lead_marker_dist_m": float(
                    np.linalg.norm(dxy - lead_targets[k][None, :], axis=1).min()
                )
                if def_idx.size
                else np.nan,
            }
        )
    return rows
