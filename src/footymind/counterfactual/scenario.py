"""Constrained tactical interventions on an observed frame ("Possession Multiverse").

An intervention relocates one or more players at a single observed frame,
optionally followed by a rule-based opponent response. The spatial analysis is
then recomputed. The engine:

1. never mutates the observed state (it works on a copy),
2. checks physical feasibility: was the new position reachable from where
   the player actually was ``horizon_s`` earlier, at a plausible speed?
3. labels everything it changes as ``HYPOTHETICAL`` (user-specified) or
   ``SIMULATED`` (produced by a response policy),
4. runs a parameter-sensitivity ensemble so the user sees how robust the
   observed-vs-hypothetical difference is to the model's assumptions.

What it does **not** do: identify causal effects. The output is a
model-based "what the spatial model says would change" comparison under
explicit assumptions. It is not evidence of what would have happened in the
match.

Opponent responses live in ``counterfactual/response.py`` (rule-based and learned policies,
limited by the kinematic envelopes of ``counterfactual/kinematics.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from footymind.counterfactual.kinematics import default_limits, reachability_envelope
from footymind.counterfactual.response import (
    MIN_SEPARATION_M,
    PITCH_MARGIN_M,
    Intervention,
    Movement,
    PolicyInfo,
    ResponseContext,
    ResponsePolicy,
    apply_response,
    policy_info,
)
from footymind.counterfactual.response_innovation import load_innovation_model
from footymind.counterfactual.response_model import load_response_model
from footymind.data.store import load_match
from footymind.spatial.analysis import FrameAnalysis, PassOption, analyze, summary
from footymind.spatial.option_model import load_default, provider_of
from footymind.spatial.pitch_control import (
    DEFAULT_GRID,
    DEFAULT_PARAMS,
    SENSITIVITY_RANGES,
    PitchControlParams,
)
from footymind.spatial.state import FrameState, causal_velocities, frame_state
from footymind.tracking.kinematics import velocity
from footymind.tracking.model import Provenance

F = npt.NDArray[np.float64]

REACH_SPEED_MPS = 7.0  # sustained high-intensity running speed for reachability
# What decides ``supported``: "disc" (constant speed REACH_SPEED_MPS from the position
# horizon_s earlier, the served contract, drawn by the web lab) or "envelope" (the kinematic
# envelope in force). Envelope v2 contains as many real positions as the disc and rejects far
# more implausible ones (docs/research/simulation_session4.md), but switching changes the API
# contract (the drawn reach disc, the schema text), so it is left to the API owner.
SUPPORTED_REACHABILITY: Literal["disc", "envelope"] = "disc"
# Below this absolute change in probability, a delta is reported but no sign-robustness score.
NEGLIGIBLE_DELTA = 0.005

__all__ = [
    "MIN_SEPARATION_M",
    "PITCH_MARGIN_M",
    "Intervention",
    "Movement",
    "ResponsePolicy",
    "run_scenario",
]

VelocityMode = Literal["keep", "zero"]
ValueModel = Literal["option_model", "physics"]


@dataclass
class Feasibility:
    player_id: str
    displacement_m: float
    reachable_radius_m: float
    reachable: bool
    origin_xy: tuple[float, float] | None  # position horizon_s earlier (OBSERVED)
    inside_pitch: bool
    min_separation_m: float
    notes: list[str] = field(default_factory=list)
    kind: str = "user"  # "user" (hypothetical) or "simulated" (response-policy move)
    # Inside the kinematic envelope (reaction delay, acceleration and speed limits) from the
    # observed position and causal velocity horizon_s earlier. None if that origin is
    # unobserved. Diagnostic only: ``supported`` keeps the constant-speed disc.
    kinematic_reachable: bool | None = None

    @property
    def supported(self) -> bool:
        return self.reachable and self.inside_pitch and self.min_separation_m >= MIN_SEPARATION_M


@dataclass
class OptionDelta:
    receiver_id: str
    observed_p: float | None
    hypothetical_p: float | None
    delta: float | None
    observed_p_interval: tuple[float, float] | None  # 10th-90th percentile over param ensemble
    hypothetical_p_interval: tuple[float, float] | None
    delta_sign_agreement: float | None  # share of ensemble members agreeing with the point sign


@dataclass
class ScenarioResult:
    observed: FrameAnalysis
    hypothetical: FrameAnalysis
    feasibility: list[Feasibility]
    movements: list[Movement]
    option_deltas: list[OptionDelta]
    summary_observed: dict[str, object]
    summary_hypothetical: dict[str, object]
    ensemble_size: int
    supported: bool
    warnings: list[str]
    assumptions: list[str]
    value_model: ValueModel
    policy: PolicyInfo


ASSUMPTIONS = [
    "Pitch-control and lane models use literature parameters (Spearman 2018; reference "
    "parameterisation) and are not calibrated on this dataset.",
    "Unmoved players keep their observed positions and causal velocities at this frame.",
    "Pass options assume a straight pass at constant average speed to the receiver's "
    "current position (no lead passes, no ball height).",
    "Reachability uses a straight-line run at <= 7 m/s from the player's observed "
    "position horizon_s earlier and ignores acceleration limits and fatigue.",
    "Opponent responses (if any) follow the selected response policy, whose own assumptions "
    "are listed with it; moves made by a policy are checked for reachability, pitch bounds "
    "and separation like user moves. No policy says what a defender would have done.",
    "Uncertainty ranges: with the option model they reflect its estimation uncertainty "
    "(possession-clustered coefficient bootstrap) at default physics inputs; with the physics "
    "heuristic they reflect a plausible range of its parameters. Neither covers behavioural "
    "responses that were not simulated.",
    "Comparisons are model-based counterfactuals. They do not identify causal effects of "
    "the intervention on the real match.",
]


OPTION_MODEL_ASSUMPTION = (
    "Option values come from a logistic option model calibrated on attempted passes "
    "(leave-one-match-out evaluated). Applying it to options players did not choose "
    "extrapolates under selection bias."
)


def check_feasibility(
    match_id: str,
    state: FrameState,
    iv: Intervention,
    horizon_s: float,
    final_state: FrameState | None = None,
    kind: str = "user",
) -> Feasibility:
    """Reachability, pitch bounds and separation for one relocated player.

    Separation is measured against ``final_state`` (all user and simulated moves applied)
    when given, so two players moved onto the same spot are detected.
    """
    td = load_match(match_id).tracking
    i = state.index(iv.player_id)
    cur = state.xy[i]
    new = np.array([iv.x, iv.y])
    disp = float(np.linalg.norm(new - cur)) if not np.isnan(cur).any() else float("nan")
    notes: list[str] = []
    k = round(horizon_s * td.fps)
    row = td.index_of_frame(state.frame)
    origin = None
    reachable = False
    radius = REACH_SPEED_MPS * horizon_s
    kinematic: bool | None = None
    if row - k >= 0 and td.period[row - k] == state.period:
        o = td.xy[row - k, i].astype(np.float64)
        if not np.isnan(o).any():
            origin = (float(o[0]), float(o[1]))
            reachable = float(np.linalg.norm(new - o)) <= radius
            env = reachability_envelope(
                o,
                causal_velocities(match_id)[row - k, i],
                horizon_s,
                default_limits(provider_of(match_id)),
            )
            kinematic = env.contains(new)
            if SUPPORTED_REACHABILITY == "envelope":
                reachable = kinematic
            notes.extend(f"kinematic check: {n}" for n in env.notes)
    if origin is None:
        notes.append("no observed position at t - horizon; reachability cannot be verified")
    if np.isnan(cur).any():
        notes.append("player is not on the pitch at this frame")
    inside = (
        abs(iv.x) <= td.pitch.half_length + PITCH_MARGIN_M
        and abs(iv.y) <= td.pitch.half_width + PITCH_MARGIN_M
    )
    ref = final_state if final_state is not None else state
    others = np.delete(ref.xy, i, axis=0)
    d_others = np.linalg.norm(others - new[None, :], axis=1)
    sep = float(np.nanmin(d_others)) if np.isfinite(d_others).any() else float("inf")
    return Feasibility(
        iv.player_id, disp, radius, reachable, origin, inside, sep, notes, kind, kinematic
    )


def ball_velocity(match_id: str, frame: int) -> F | None:
    """Causal (backward 0.2 s) ball velocity at ``frame``; None if unavailable."""
    td = load_match(match_id).tracking
    row = td.index_of_frame(frame)
    k = max(1, round(0.2 * td.fps))
    if row - k < 0 or td.period[row - k] != td.period[row]:
        return None
    v = velocity(td.ball_xy[row - k : row + 1], td.fps, "causal", 0.2)[-1].astype(np.float64)
    return None if np.isnan(v).any() else v


def _apply(
    state: FrameState,
    interventions: list[Intervention],
    policy: ResponsePolicy,
    follow_fraction: float,
    velocity_mode: VelocityMode,
    ctx: ResponseContext | None = None,
) -> tuple[FrameState, list[Movement], list[str]]:
    hyp = state.copy()
    moves: list[Movement] = []
    for iv in interventions:
        i = hyp.index(iv.player_id)
        frm = (float(hyp.xy[i, 0]), float(hyp.xy[i, 1]))
        hyp.xy[i] = [iv.x, iv.y]
        hyp.position_provenance[i] = Provenance.HYPOTHETICAL
        if velocity_mode == "zero":
            hyp.vel[i] = [0.0, 0.0]
            hyp.velocity_provenance[i] = Provenance.HYPOTHETICAL
        moves.append(Movement(iv.player_id, frm, (iv.x, iv.y), Provenance.HYPOTHETICAL, "user"))
        if iv.player_id == state.player_ids[state.carrier] if state.carrier is not None else False:
            # Moving the carrier moves the ball with them.
            hyp.ball = hyp.ball + (np.array([iv.x, iv.y]) - np.array(frm))
    ctx = ctx or ResponseContext(follow_fraction=follow_fraction)
    ctx.follow_fraction = follow_fraction
    sim, notes = apply_response(state, hyp, interventions, policy, ctx)
    return hyp, moves + sim, notes


def _ensemble_params(n: int, seed: int) -> list[PitchControlParams]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        kw = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in SENSITIVITY_RANGES.items()}
        out.append(DEFAULT_PARAMS.with_(**kw))
    return out


def run_scenario(
    match_id: str,
    frame: int,
    interventions: list[Intervention],
    policy: ResponsePolicy = "none",
    follow_fraction: float = 0.5,
    velocity_mode: VelocityMode = "keep",
    horizon_s: float = 1.5,
    ensemble: int = 24,
    seed: int = 0,
    value_model: ValueModel = "option_model",
) -> ScenarioResult:
    if not interventions:
        raise ValueError("at least one intervention is required")
    observed_state = frame_state(match_id, frame)
    if observed_state.attacking_team is None:
        raise ValueError("no possession at this frame")
    for iv in interventions:
        if iv.player_id not in observed_state.player_ids:
            raise KeyError(iv.player_id)
        if np.isnan(observed_state.xy[observed_state.index(iv.player_id)]).any():
            raise ValueError(f"{iv.player_id} is not on the pitch at frame {frame}")
    td = load_match(match_id).tracking
    ctx = ResponseContext(
        pitch=td.pitch,
        horizon_s=horizon_s,
        limits=default_limits(provider_of(match_id)),
        follow_fraction=follow_fraction,
        ball_velocity=ball_velocity(match_id, frame) if policy == "learned" else None,
        model=load_response_model(provider_of(match_id)) if policy == "learned" else None,
        innovation_model=(
            load_innovation_model(provider_of(match_id)) if policy == "learned_innovation" else None
        ),
    )
    hyp_state, moves, policy_notes = _apply(
        observed_state, interventions, policy, follow_fraction, velocity_mode, ctx
    )
    feas = [
        check_feasibility(match_id, observed_state, iv, horizon_s, final_state=hyp_state)
        for iv in interventions
    ]
    # Moves made by the response policy must be physically plausible too.
    feas += [
        check_feasibility(
            match_id,
            observed_state,
            Intervention(m.player_id, m.to_xy[0], m.to_xy[1]),
            horizon_s,
            final_state=hyp_state,
            kind="simulated",
        )
        for m in moves
        if m.provenance == Provenance.SIMULATED
    ]

    obs_a = analyze(observed_state, DEFAULT_PARAMS, DEFAULT_GRID)
    hyp_a = analyze(hyp_state, DEFAULT_PARAMS, DEFAULT_GRID)

    params_list = _ensemble_params(ensemble, seed)

    def val(o: PassOption) -> float:
        if value_model == "option_model" and o.p_success_model is not None:
            return o.p_success_model
        return o.p_success

    obs_ens: dict[str, list[float]] = {}
    hyp_ens: dict[str, list[float]] = {}
    # Each ensemble member pairs one pitch-control parameter draw with one option-model
    # bootstrap member, so both model-parameter and estimation uncertainty are sampled.
    # Option model: sample its estimation uncertainty (coefficient bootstrap) with the physics
    # inputs at the parameters it was trained with. Perturbing the physics parameters would feed
    # it out-of-distribution features (independent review). Physics value model: sample the
    # physics-parameter ensemble.
    for member, p in enumerate(params_list):
        if value_model == "option_model":
            p = DEFAULT_PARAMS
        for o in analyze(observed_state, p, grid=None, model_member=member).pass_options:
            obs_ens.setdefault(o.receiver_id, []).append(val(o))
        for o in analyze(hyp_state, p, grid=None, model_member=member).pass_options:
            hyp_ens.setdefault(o.receiver_id, []).append(val(o))

    obs_map = {o.receiver_id: val(o) for o in obs_a.pass_options}
    hyp_map = {o.receiver_id: val(o) for o in hyp_a.pass_options}
    if value_model == "option_model" and load_default(match_id) is None:
        value_model = "physics"
    deltas = []
    for rid in sorted(set(obs_map) | set(hyp_map)):
        op, hp = obs_map.get(rid), hyp_map.get(rid)
        delta = hp - op if op is not None and hp is not None else None
        oe, he = np.array(obs_ens.get(rid, [])), np.array(hyp_ens.get(rid, []))
        agree = None
        if delta is not None and oe.size == he.size and oe.size > 0:
            d_ens = he - oe
            agree = (
                float(np.mean(np.sign(d_ens) == np.sign(delta)))
                if abs(delta) >= NEGLIGIBLE_DELTA
                else None
            )
        deltas.append(
            OptionDelta(
                receiver_id=rid,
                observed_p=op,
                hypothetical_p=hp,
                delta=delta,
                observed_p_interval=(
                    (float(np.percentile(oe, 10)), float(np.percentile(oe, 90)))
                    if oe.size
                    else None
                ),
                hypothetical_p_interval=(
                    (float(np.percentile(he, 10)), float(np.percentile(he, 90)))
                    if he.size
                    else None
                ),
                delta_sign_agreement=agree,
            )
        )
    deltas.sort(key=lambda d: -abs(d.delta or 0.0))

    warnings = list(dict.fromkeys(obs_a.warnings + hyp_a.warnings))
    env_text = default_limits(provider_of(match_id)).describe()
    warnings.extend(f"response policy {n}" for n in policy_notes)
    for f in feas:
        who = f"{f.player_id} (simulated response)" if f.kind == "simulated" else f.player_id
        if not f.reachable:
            region = (
                f"at {REACH_SPEED_MPS:.0f} m/s"
                if SUPPORTED_REACHABILITY == "disc"
                else f"within the kinematic envelope ({env_text})"
            )
            warnings.append(
                f"{who}: target is outside the region reachable in {horizon_s:.1f}s "
                f"{region}; scenario is physically unsupported"
            )
        if not f.inside_pitch:
            warnings.append(f"{who}: target is outside the pitch")
        if f.min_separation_m < MIN_SEPARATION_M:
            warnings.append(f"{who}: target overlaps another player ({f.min_separation_m:.2f} m)")
        warnings.extend(f"{who}: {n}" for n in f.notes)
    unstable = [
        d.receiver_id
        for d in deltas
        if d.delta_sign_agreement is not None
        and abs(d.delta or 0) > 0.02
        and d.delta_sign_agreement < 0.8
    ]
    if unstable:
        warnings.append(
            "direction of change is sensitive to model parameters for: " + ", ".join(unstable)
        )
    return ScenarioResult(
        observed=obs_a,
        hypothetical=hyp_a,
        feasibility=feas,
        movements=moves,
        option_deltas=deltas,
        summary_observed=summary(obs_a),
        summary_hypothetical=summary(hyp_a),
        ensemble_size=ensemble,
        supported=all(f.supported for f in feas),
        warnings=warnings,
        assumptions=ASSUMPTIONS
        + ([OPTION_MODEL_ASSUMPTION] if value_model == "option_model" else []),
        value_model=value_model,
        policy=policy_info(policy, provider_of(match_id)),
    )
