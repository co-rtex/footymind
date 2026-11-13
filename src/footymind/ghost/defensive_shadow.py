"""Defensive Shadow: how much a defender's actual position suppresses passing options.

Definition (v1, ESTIMATED; not validated against outcomes yet)
---------------------------------------------------------------
For an observed frame with an attacking team in possession, define the team's option
value under a value model (by default the calibrated option model):

* ``V_best(s)``: the highest success probability over the carrier's pass options;
* ``V_sum(s)``: the sum of success probabilities over options (the "expected number of
  completable options").

For defender ``d``, sample alternative positions ``a_k``: ``n_directions`` points on a ring
centred on the defender's *current* observed position, with radius
``ring_fraction * 7 m/s * horizon_s`` (3.5 m by default), clipped to the pitch. The radius
is a plausible-displacement scale. Unlike the scenario engine, the alternatives are **not**
checked against where the defender actually was ``horizon_s`` earlier. The defensive
shadow is

    shadow_best(d) = mean_k V_best(s with d at a_k) - V_best(s observed)

and likewise for ``V_sum``. Positive values mean that, according to the model, the
defender's actual position removes more passing value than the typical nearby
alternative.

What this is not: a causal effect, a learned defensive policy, or a validated rating.
Other players are held fixed (no response), and the value model extrapolates to
unchosen options (selection bias, see docs/RESEARCH.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from footymind.data.store import load_match
from footymind.spatial.analysis import pass_options
from footymind.spatial.pitch_control import DEFAULT_PARAMS, PitchControlParams
from footymind.spatial.state import FrameState, frame_state

REACH_SPEED_MPS = 7.0
# |shadow_sum| below this is not distinguishable from tracking noise: the mean absolute
# change under sigma = 0.3 m position noise was 0.025 (experiments/results/shadow_stability_v1).
NOISE_FLOOR = 0.025
ValueModel = Literal["option_model", "physics"]


@dataclass
class DefenderShadow:
    player_id: str
    x: float
    y: float
    shadow_best: float
    shadow_sum: float
    alternatives: list[tuple[float, float]]
    alt_radius_m: float
    most_suppressed_receiver: str | None  # option whose value rises most when d moves away
    most_suppressed_delta: float


@dataclass
class ShadowResult:
    match_id: str
    frame: int
    attacking_team: str
    carrier_id: str | None
    value_model: ValueModel
    v_best_observed: float
    v_sum_observed: float
    defenders: list[DefenderShadow]
    assumptions: list[str]


ASSUMPTIONS = [
    "Defensive shadow compares the observed defender position with nearby positions the "
    "defender could plausibly have occupied (ring at a fraction of the reachable radius).",
    "All other players stay at their observed positions; no response is simulated.",
    "Option values come from the selected value model and extrapolate to unchosen options.",
    "This is a model-based estimate, not a validated defensive rating or a causal effect.",
]


def _values(state: FrameState, model: ValueModel, params: PitchControlParams) -> dict[str, float]:
    out = {}
    for o in pass_options(state, params):
        v = (
            o.p_success_model
            if model == "option_model" and o.p_success_model is not None
            else (o.p_success)
        )
        out[o.receiver_id] = float(v)
    return out


def defensive_shadow(
    match_id: str,
    frame: int,
    horizon_s: float = 1.0,
    ring_fraction: float = 0.5,
    n_directions: int = 8,
    value_model: ValueModel = "option_model",
    params: PitchControlParams = DEFAULT_PARAMS,
) -> ShadowResult:
    """Defensive shadow at an observed frame (see module docstring)."""
    st = frame_state(match_id, frame)
    pitch = load_match(match_id).tracking.pitch
    return shadow_for_state(
        st,
        pitch.half_length,
        pitch.half_width,
        horizon_s,
        ring_fraction,
        n_directions,
        value_model,
        params,
    )


def shadow_for_state(
    st: FrameState,
    half_length: float = 52.5,
    half_width: float = 34.0,
    horizon_s: float = 1.0,
    ring_fraction: float = 0.5,
    n_directions: int = 8,
    value_model: ValueModel = "option_model",
    params: PitchControlParams = DEFAULT_PARAMS,
) -> ShadowResult:
    """Pure computation on a ``FrameState``, so constructed states can be property-tested."""
    if st.attacking_team is None:
        raise ValueError("no possession at this frame")
    if np.isnan(st.ball).any():
        raise ValueError("ball not observed at this frame")
    base = _values(st, value_model, params)
    if not base:
        raise ValueError("no pass options at this frame")
    vb0, vs0 = max(base.values()), sum(base.values())
    radius = REACH_SPEED_MPS * horizon_s * ring_fraction
    angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
    out: list[DefenderShadow] = []
    dfn = (~st.team_mask(st.attacking_team)) & st.on_pitch()
    for d in np.flatnonzero(dfn):
        if st.is_gk[d]:
            continue  # goalkeepers handle the ball; ring displacement is not meaningful
        x0, y0 = st.xy[d]
        alts = []
        vb, vs = [], []
        gains: dict[str, list[float]] = {}
        for a in angles:
            ax = float(np.clip(x0 + radius * np.cos(a), -half_length, half_length))
            ay = float(np.clip(y0 + radius * np.sin(a), -half_width, half_width))
            alt = st.copy()
            alt.xy[d] = [ax, ay]
            vals = _values(alt, value_model, params)
            alts.append((round(ax, 2), round(ay, 2)))
            vb.append(max(vals.values()))
            vs.append(sum(vals.values()))
            for rid, v in vals.items():
                gains.setdefault(rid, []).append(v - base.get(rid, v))
        mean_gain = {rid: float(np.mean(g)) for rid, g in gains.items()}
        top = max(mean_gain, key=lambda r: mean_gain[r]) if mean_gain else None
        out.append(
            DefenderShadow(
                player_id=st.player_ids[d],
                x=float(x0),
                y=float(y0),
                shadow_best=float(np.mean(vb) - vb0),
                shadow_sum=float(np.mean(vs) - vs0),
                alternatives=alts,
                alt_radius_m=radius,
                most_suppressed_receiver=top,
                most_suppressed_delta=mean_gain[top] if top else 0.0,
            )
        )
    out.sort(key=lambda s: -s.shadow_sum)
    return ShadowResult(
        match_id=st.match_id,
        frame=st.frame,
        attacking_team=st.attacking_team,
        carrier_id=st.player_ids[st.carrier] if st.carrier is not None else None,
        value_model=value_model,
        v_best_observed=vb0,
        v_sum_observed=vs0,
        defenders=out,
        assumptions=ASSUMPTIONS,
    )
