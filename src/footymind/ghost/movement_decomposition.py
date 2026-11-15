"""Who changed the passing picture: attackers' or defenders' off-ball movement?

Definition (session 3, pre-specified, ESTIMATED, model-based)
-------------------------------------------------------------
At an observed frame ``t`` with a team in possession, let ``V(A, D)`` be the team's passing
opportunity under the serving option model. Three measures are reported:

* ``forward`` (primary): the best completion probability among *progressive* options,
  teammates more than ``PROGRESSIVE_M`` ahead of the ball (0 if there is none). The plain
  best option is dominated by safe backward passes (often p ≈ 1), which no forward run can
  change. This was found on a constructed test case before any match data was analysed.
* ``sum``: summed completion probability over all teammates;
* ``any``: the best option overall (kept to document the saturation).

 Here ``A`` holds the off-ball attackers' positions and
velocities and ``D`` the defenders'. The carrier and the ball are fixed at ``t`` in every
state, so only off-ball movement varies. Each side is placed either at ``t`` (index 1) or
``horizon_s`` earlier (index 0):

* attack effect  = ½[(V10 − V00) + (V11 − V01)]   (Shapley value of the attackers' movement)
* defense effect = ½[(V01 − V00) + (V11 − V10)]
* interaction    = V11 − V10 − V01 + V00

The two effects sum to ``V11 − V00``. A large interaction means the value of one side's
movement depends on the other side's. Crediting attackers while holding defenders at their
observed positions, which is what ``run_value`` does, is then not well defined without a
model of how defenders would have responded.

What it is not: a causal decomposition. Both sides' "earlier" positions are real
observations, but the mixed states (V10, V01) never happened, and the option model
extrapolates to them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from footymind.spatial.analysis import pass_options
from footymind.spatial.pitch_control import DEFAULT_PARAMS, PitchControlParams
from footymind.spatial.state import FrameState
from footymind.tracking.model import Provenance

PROGRESSIVE_M = 5.0
MEASURES = ("forward", "sum", "any")


@dataclass(frozen=True)
class Effects:
    attack: float
    defense: float
    interaction: float


@dataclass(frozen=True)
class Decomposition:
    v: dict[str, dict[str, float]]  # measure -> {"11", "10", "01", "00"} -> value
    effects: dict[str, Effects]  # measure -> Shapley effects
    n_attackers_moved: int
    n_defenders_moved: int

    @property
    def attack_effect(self) -> float:
        return self.effects["forward"].attack

    @property
    def defense_effect(self) -> float:
        return self.effects["forward"].defense

    @property
    def interaction(self) -> float:
        return self.effects["forward"].interaction


def _opportunity(st: FrameState, params: PitchControlParams) -> dict[str, float]:
    opts = pass_options(st, params)
    vals = [o.p_success_model if o.p_success_model is not None else o.p_success for o in opts]
    fwd = [v for v, o in zip(vals, opts, strict=True) if o.forward_progress_m > PROGRESSIVE_M]
    return {
        "forward": float(max(fwd)) if fwd else 0.0,
        "sum": float(sum(vals)) if vals else np.nan,
        "any": float(max(vals)) if vals else np.nan,
    }


def _shapley(v: dict[str, float]) -> tuple[float, float, float]:
    att = 0.5 * ((v["10"] - v["00"]) + (v["11"] - v["01"]))
    dfn = 0.5 * ((v["01"] - v["00"]) + (v["11"] - v["10"]))
    inter = v["11"] - v["10"] - v["01"] + v["00"]
    return att, dfn, inter


def decompose(
    st: FrameState,
    earlier_xy: np.ndarray,
    earlier_vel: np.ndarray,
    params: PitchControlParams = DEFAULT_PARAMS,
) -> Decomposition:
    """Decompose the change in team opportunity over the window into attack/defense parts.

    Players with no earlier position keep their position at ``t`` in every state (they count
    as not moved). The carrier never moves.
    """
    if st.attacking_team is None or np.isnan(st.ball).any() or st.carrier is None:
        raise ValueError("needs a possession, a carrier and an observed ball")
    on = st.on_pitch()
    has_earlier = ~np.isnan(earlier_xy).any(axis=1)
    att = st.team_mask(st.attacking_team) & on & has_earlier
    att[st.carrier] = False
    dfn = (~st.team_mask(st.attacking_team)) & on & has_earlier

    def state(a_now: bool, d_now: bool) -> FrameState:
        s = st.copy()
        for mask, now in ((att, a_now), (dfn, d_now)):
            if not now:
                idx = np.flatnonzero(mask)
                s.xy[idx] = earlier_xy[idx]
                s.vel[idx] = np.nan_to_num(earlier_vel[idx])
                for i in idx:
                    s.position_provenance[i] = Provenance.HYPOTHETICAL
        return s

    by_state = {
        key: _opportunity(state(a_now, d_now), params)
        for key, (a_now, d_now) in {
            "11": (True, True),
            "10": (True, False),
            "01": (False, True),
            "00": (False, False),
        }.items()
    }
    v = {m: {k: by_state[k][m] for k in by_state} for m in MEASURES}
    return Decomposition(
        v=v,
        effects={m: Effects(*_shapley(v[m])) for m in MEASURES},
        n_attackers_moved=int(att.sum()),
        n_defenders_moved=int(dfn.sum()),
    )
