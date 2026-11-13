"""Frame-level tactical analysis: pitch control, pass options, pressure.

Every output field is labelled with a ``Provenance`` so that the API and UI
can distinguish observed, derived, estimated and hypothetical quantities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from footymind.spatial.lanes import lane
from footymind.spatial.pitch_control import (
    DEFAULT_GRID,
    DEFAULT_PARAMS,
    Grid,
    PitchControlParams,
    pitch_control,
)
from footymind.spatial.state import FrameState
from footymind.tracking.model import Provenance

F = npt.NDArray[np.float64]


@dataclass
class PassOption:
    receiver_id: str
    target: tuple[float, float]
    distance_m: float
    forward_progress_m: float  # +ve = towards the opponent goal
    ball_travel_time_s: float
    p_control_at_target: float  # ESTIMATED: PPCF of attacking team at receiver location
    p_lane_clear: float  # ESTIMATED: heuristic
    p_success: float  # ESTIMATED: p_control_at_target * p_lane_clear (heuristic, uncalibrated)
    lane_min_margin_s: float  # DERIVED
    lane_closest_defender_id: str | None
    lane_closest_defender_perp_m: float  # DERIVED
    defenders_within_3m_of_lane: int  # DERIVED


@dataclass
class Pressure:
    nearest_defender_id: str | None
    nearest_defender_distance_m: float
    closing_speed_mps: float  # component of relative velocity reducing the gap. NaN if unknown
    defenders_within_5m: int
    defenders_within_10m: int


@dataclass
class FrameAnalysis:
    state: FrameState
    params: PitchControlParams
    grid: Grid | None
    control_grid: F | None  # (ny*nx,) attacking-team control probability
    pass_options: list[PassOption]
    pressure: Pressure | None
    control_area_m2: float | None  # attacking team, whole pitch
    control_area_final_third_m2: float | None
    warnings: list[str] = field(default_factory=list)

    @property
    def provenance(self) -> dict[str, Provenance]:
        return {
            "positions": Provenance.OBSERVED,
            "velocities": Provenance.DERIVED,
            "pitch_control": Provenance.ESTIMATED,
            "pass_options.p_*": Provenance.ESTIMATED,
            "pass_options.lane_*": Provenance.DERIVED,
            "pressure": Provenance.DERIVED,
        }


def _teams(state: FrameState) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    if state.attacking_team is None:
        raise ValueError("no attacking team at this frame")
    att = state.team_mask(state.attacking_team)
    return att, ~att


def analyze(
    state: FrameState,
    params: PitchControlParams = DEFAULT_PARAMS,
    grid: Grid | None = DEFAULT_GRID,
) -> FrameAnalysis:
    warnings: list[str] = []
    if state.attacking_team is None:
        return FrameAnalysis(state, params, None, None, [], None, None, None, ["no possession"])
    att, dfn = _teams(state)
    on = state.on_pitch()
    missing_v = [
        state.player_ids[i]
        for i in range(len(state.player_ids))
        if on[i] and np.isnan(state.vel[i, 0])
    ]
    if missing_v:
        warnings.append(
            f"velocity unavailable for {len(missing_v)} player(s) "
            f"({', '.join(missing_v[:4])}{'...' if len(missing_v) > 4 else ''}); treated as 0 m/s"
        )
    if np.isnan(state.ball).any():
        warnings.append("ball position not observed at this frame; pass options unavailable")

    ap, av = state.xy[att], state.vel[att]
    dp, dv = state.xy[dfn], state.vel[dfn]
    a_gk, d_gk = state.is_gk[att], state.is_gk[dfn]

    control = None
    area = area_ft = None
    if grid is not None and not np.isnan(state.ball).any():
        centres = grid.centres()
        control, _, _ = pitch_control(centres, state.ball, ap, av, dp, dv, a_gk, d_gk, params)
        cell = grid.dx * grid.dy
        area = float(np.nansum(control) * cell)
        third = centres[:, 0] * state.attack_sign > grid.length / 6
        area_ft = float(np.nansum(control[third]) * cell)

    options = pass_options(state, params) if not np.isnan(state.ball).any() else []
    if state.carrier is None and not np.isnan(state.ball).any():
        warnings.append(
            "no attacking player within 1.5 m of the ball (ball in flight or loose); "
            "pass options are computed from the ball position"
        )
    return FrameAnalysis(
        state=state,
        params=params,
        grid=grid,
        control_grid=control,
        pass_options=options,
        pressure=pressure(state),
        control_area_m2=area,
        control_area_final_third_m2=area_ft,
        warnings=warnings,
    )


def pass_options(
    state: FrameState, params: PitchControlParams = DEFAULT_PARAMS
) -> list[PassOption]:
    att, dfn = _teams(state)
    on = state.on_pitch()
    receivers = [i for i in np.flatnonzero(att & on) if i != state.carrier]
    if not receivers:
        return []
    targets = state.xy[receivers]
    ppcf, _, _ = pitch_control(
        targets,
        state.ball,
        state.xy[att],
        state.vel[att],
        state.xy[dfn],
        state.vel[dfn],
        state.is_gk[att],
        state.is_gk[dfn],
        params,
    )
    def_ids = [state.player_ids[i] for i in np.flatnonzero(dfn)]
    out = []
    for k, i in enumerate(receivers):
        tgt = state.xy[i]
        lr = lane(state.ball, tgt, state.xy[dfn], state.vel[dfn], params)
        p_ctrl = float(ppcf[k])
        out.append(
            PassOption(
                receiver_id=state.player_ids[i],
                target=(float(tgt[0]), float(tgt[1])),
                distance_m=lr.length_m,
                forward_progress_m=float((tgt[0] - state.ball[0]) * state.attack_sign),
                ball_travel_time_s=lr.ball_travel_time_s,
                p_control_at_target=p_ctrl,
                p_lane_clear=lr.p_lane_clear,
                p_success=p_ctrl * lr.p_lane_clear,
                lane_min_margin_s=lr.min_margin_s,
                lane_closest_defender_id=(
                    def_ids[lr.closest_defender_index]
                    if lr.closest_defender_index is not None
                    else None
                ),
                lane_closest_defender_perp_m=lr.closest_defender_perp_m,
                defenders_within_3m_of_lane=lr.n_defenders_within_3m_of_lane,
            )
        )
    out.sort(key=lambda o: -o.p_success)
    return out


def pressure(state: FrameState) -> Pressure | None:
    if state.carrier is None or state.attacking_team is None:
        return None
    _, dfn = _teams(state)
    c = state.carrier
    idx = np.flatnonzero(dfn & state.on_pitch())
    if idx.size == 0:
        return None
    rel = state.xy[idx] - state.xy[c][None, :]
    d = np.linalg.norm(rel, axis=1)
    j = int(np.argmin(d))
    dj = idx[j]
    closing = np.nan
    if not (np.isnan(state.vel[dj]).any() or np.isnan(state.vel[c]).any()) and d[j] > 0:
        unit = rel[j] / d[j]  # carrier -> defender
        closing = float(-(state.vel[dj] - state.vel[c]) @ unit)
    return Pressure(
        nearest_defender_id=state.player_ids[dj],
        nearest_defender_distance_m=float(d[j]),
        closing_speed_mps=closing,
        defenders_within_5m=int((d < 5).sum()),
        defenders_within_10m=int((d < 10).sum()),
    )


def summary(a: FrameAnalysis) -> dict[str, Any]:
    """Compact scalar summary (used for deltas and sensitivity analysis)."""
    best = a.pass_options[0] if a.pass_options else None
    return {
        "control_area_m2": a.control_area_m2,
        "control_area_final_third_m2": a.control_area_final_third_m2,
        "best_option_receiver": best.receiver_id if best else None,
        "best_option_p_success": best.p_success if best else None,
        "n_options_p_over_0_5": sum(o.p_success > 0.5 for o in a.pass_options),
    }
