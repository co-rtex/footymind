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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from footymind.data.store import load_match
from footymind.spatial.analysis import FrameAnalysis, PassOption, analyze, summary
from footymind.spatial.option_model import load_default
from footymind.spatial.pitch_control import (
    DEFAULT_GRID,
    DEFAULT_PARAMS,
    SENSITIVITY_RANGES,
    PitchControlParams,
)
from footymind.spatial.state import FrameState, frame_state
from footymind.tracking.model import Provenance

F = npt.NDArray[np.float64]

REACH_SPEED_MPS = 7.0  # sustained high-intensity running speed for reachability
MIN_SEPARATION_M = 0.5
# Below this absolute change in probability, a delta is reported but no sign-robustness score.
NEGLIGIBLE_DELTA = 0.005
PITCH_MARGIN_M = 2.0

ResponsePolicy = Literal["none", "marker_follows"]
VelocityMode = Literal["keep", "zero"]
ValueModel = Literal["option_model", "physics"]


@dataclass
class Intervention:
    player_id: str
    x: float
    y: float


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

    @property
    def supported(self) -> bool:
        return self.reachable and self.inside_pitch and self.min_separation_m >= MIN_SEPARATION_M


@dataclass
class Movement:
    player_id: str
    from_xy: tuple[float, float]
    to_xy: tuple[float, float]
    provenance: Provenance
    reason: str


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


ASSUMPTIONS = [
    "Pitch-control and lane models use literature parameters (Spearman 2018; reference "
    "parameterisation) and are not calibrated on this dataset.",
    "Unmoved players keep their observed positions and causal velocities at this frame.",
    "Pass options assume a straight pass at constant average speed to the receiver's "
    "current position (no lead passes, no ball height).",
    "Reachability uses a straight-line run at <= 7 m/s from the player's observed "
    "position horizon_s earlier and ignores acceleration limits and fatigue.",
    "Opponent response policies are simple rules, not learned behaviour models; moves made by "
    "a policy are checked for reachability, pitch bounds and separation like user moves.",
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
    if row - k >= 0 and td.period[row - k] == state.period:
        o = td.xy[row - k, i].astype(np.float64)
        if not np.isnan(o).any():
            origin = (float(o[0]), float(o[1]))
            reachable = float(np.linalg.norm(new - o)) <= radius
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
    return Feasibility(iv.player_id, disp, radius, reachable, origin, inside, sep, notes, kind)


def _apply(
    state: FrameState,
    interventions: list[Intervention],
    policy: ResponsePolicy,
    follow_fraction: float,
    velocity_mode: VelocityMode,
) -> tuple[FrameState, list[Movement]]:
    hyp = state.copy()
    moves: list[Movement] = []
    moved_ids = {iv.player_id for iv in interventions}
    shifted: set[str] = set()
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
    if policy == "marker_follows":
        # Rule: when an attacker is relocated, the defender nearest to the
        # attacker's observed position (within 10 m) shifts by a fraction of the
        # same displacement. Relocated defenders trigger no response.
        for iv in interventions:
            i = state.index(iv.player_id)
            if state.teams[i] != state.attacking_team:
                continue
            opp = ~state.team_mask(state.teams[i]) & state.on_pitch()
            for mid in moved_ids | shifted:
                opp[state.index(mid)] = False  # never shift a player twice
            if not opp.any() or np.isnan(state.xy[i]).any():
                continue
            d = np.linalg.norm(state.xy - state.xy[i][None, :], axis=1)
            d[~opp] = np.inf
            j = int(np.argmin(d))
            if d[j] > 10.0:
                continue
            shift = (np.array([iv.x, iv.y]) - state.xy[i]) * follow_fraction
            frm = (float(hyp.xy[j, 0]), float(hyp.xy[j, 1]))
            hyp.xy[j] = hyp.xy[j] + shift
            hyp.position_provenance[j] = Provenance.SIMULATED
            shifted.add(state.player_ids[j])
            moves.append(
                Movement(
                    state.player_ids[j],
                    frm,
                    (float(hyp.xy[j, 0]), float(hyp.xy[j, 1])),
                    Provenance.SIMULATED,
                    f"marker of {iv.player_id} follows {follow_fraction:.0%} of the displacement "
                    "(rule-based response)",
                )
            )
    return hyp, moves


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
    hyp_state, moves = _apply(observed_state, interventions, policy, follow_fraction, velocity_mode)
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
    if value_model == "option_model" and load_default() is None:
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
    for f in feas:
        who = f"{f.player_id} (simulated response)" if f.kind == "simulated" else f.player_id
        if not f.reachable:
            warnings.append(
                f"{who}: target is outside the region reachable in {horizon_s:.1f}s "
                f"at {REACH_SPEED_MPS:.0f} m/s; scenario is physically unsupported"
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
    )
