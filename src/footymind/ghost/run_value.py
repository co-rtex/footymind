"""Run value: how much an attacker's recent off-ball movement changed their availability.

Definition (v1, ESTIMATED, experimental; pre-specified before validation)
-------------------------------------------------------------------------
At an observed frame ``t`` with a team in possession, for every attacker ``a`` other than
the carrier:

* counterfactual state: ``a`` alone is placed at their **observed** position and causal
  velocity from ``horizon_s`` earlier ("had they not made the last ``horizon_s`` of
  movement"). Everyone else, defenders included, keeps their observed state at ``t``. The
  defenders' positions already contain whatever response the run provoked.
* ``own_run_value(a) = p_a(observed) - p_a(counterfactual)``, where ``p_a`` is the
  option-model probability of completing a pass to ``a``.
* ``team_run_value(a)``: the same difference for the sum of all *other* options. Through
  pitch control the runner's position also affects teammates' options, but no defender
  response is simulated, so this part is expected to be small.

What it is not: a causal effect of the run, or a valuation of space created by dragging
defenders. Defender responses are not simulated. The option model also extrapolates to
counterfactual positions (selection bias, see docs/RESEARCH.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from footymind.data.store import load_match
from footymind.spatial.analysis import pass_options
from footymind.spatial.pitch_control import DEFAULT_PARAMS, PitchControlParams
from footymind.spatial.state import FrameState, causal_velocities, frame_state
from footymind.tracking.model import Provenance


@dataclass
class RunValue:
    player_id: str
    own_run_value: float
    team_run_value: float
    p_observed: float
    p_counterfactual: float
    displacement_m: float  # distance moved in the last horizon_s (observed)
    speed_mps: float | None


def _values(st: FrameState, params: PitchControlParams) -> dict[str, float]:
    return {
        o.receiver_id: (o.p_success_model if o.p_success_model is not None else o.p_success)
        for o in pass_options(st, params)
    }


def run_values_for_state(
    st: FrameState,
    earlier_xy: np.ndarray,
    earlier_vel: np.ndarray,
    params: PitchControlParams = DEFAULT_PARAMS,
) -> list[RunValue]:
    """Run values given the full observed state and every player's state ``horizon_s`` earlier."""
    if st.attacking_team is None or np.isnan(st.ball).any():
        raise ValueError("needs a possession and an observed ball")
    base = _values(st, params)
    out: list[RunValue] = []
    att = st.team_mask(st.attacking_team) & st.on_pitch()
    for a in np.flatnonzero(att):
        pid = st.player_ids[a]
        if a == st.carrier or pid not in base or np.isnan(earlier_xy[a]).any():
            continue
        cf = st.copy()
        cf.xy[a] = earlier_xy[a]
        cf.vel[a] = np.nan_to_num(earlier_vel[a])
        cf.position_provenance[a] = Provenance.HYPOTHETICAL
        vals = _values(cf, params)
        if pid not in vals:
            continue
        others_obs = sum(v for k, v in base.items() if k != pid)
        others_cf = sum(v for k, v in vals.items() if k != pid)
        v = st.vel[a]
        out.append(
            RunValue(
                player_id=pid,
                own_run_value=base[pid] - vals[pid],
                team_run_value=others_obs - others_cf,
                p_observed=base[pid],
                p_counterfactual=vals[pid],
                displacement_m=float(np.linalg.norm(st.xy[a] - earlier_xy[a])),
                speed_mps=None if np.isnan(v).any() else float(np.linalg.norm(v)),
            )
        )
    out.sort(key=lambda r: -r.own_run_value)
    return out


def run_values(match_id: str, frame: int, horizon_s: float = 1.0) -> list[RunValue]:
    td = load_match(match_id).tracking
    st = frame_state(match_id, frame)
    row = td.index_of_frame(frame)
    k = round(horizon_s * td.fps)
    if row - k < 0 or td.period[row - k] != st.period:
        raise ValueError("no observed state horizon_s earlier in this period")
    earlier_xy = td.xy[row - k].astype(np.float64)
    earlier_vel = causal_velocities(match_id)[row - k].astype(np.float64)
    return run_values_for_state(st, earlier_xy, earlier_vel)
