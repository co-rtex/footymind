"""Opponent response policies for the Possession Multiverse.

After a user intervention (HYPOTHETICAL moves), a response policy may move **defenders**
(the team out of possession). Every move it makes is labelled ``SIMULATED`` and carries a
human-readable reason. Policies never move attackers, goalkeepers or user-moved players,
and skip players without a position.

Policies
--------
* ``none``: no response.
* ``marker_follows``: the legacy rule (nearest defender within 10 m of a relocated attacker
  shifts by ``follow_fraction`` of the displacement). Kept bit-for-bit; no kinematic limit.
* ``press_carrier``, ``zonal_shift``, ``lane_cover``: interpretable rules.
* ``learned``: a ridge model of short-horizon defender displacement fit on real tracking
  (``response_model.py``).

Differential responses (all policies except ``none`` and ``marker_follows``)
----------------------------------------------------------------------------
A policy maps a game state ``S`` to a desired position ``g_j(S)`` for defender ``j`` after
the horizon. That target is projected onto ``j``'s kinematic envelope ``E_j`` (built from
``j``'s observed position and causal velocity, see ``kinematics.py``), and the move applied
to the analysed frame is the **difference**::

    delta_j = clip_Ej(g_j(S_hypothetical)) - clip_Ej(g_j(S_observed))

So a defender moves only by the change in the policy's target that the intervention causes,
limited by what is physically reachable. Without an intervention nothing moves. The
analysed instant stays the observed frame, and the response is stable: projection onto a
convex set is non-expansive, so a small intervention gives a small response.

What this is not: a prediction of what a defender "would have done". The rules are a priori
assumptions, and the learned model forecasts the observed evolution of play. No ground truth
exists for responses to hypothetical interventions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from footymind.counterfactual.kinematics import (
    DEFAULT_LIMITS,
    Envelope,
    KinematicLimits,
    reachability_envelope,
)
from footymind.counterfactual.response_model import (
    PROVIDER_RESOURCES,
    ResponseModel,
    features,
    load_response_model,
)
from footymind.spatial.state import FrameState
from footymind.tracking.model import PitchSpec, Provenance

F = npt.NDArray[np.float64]

ResponsePolicy = Literal[
    "none", "marker_follows", "press_carrier", "zonal_shift", "lane_cover", "learned"
]
POLICY_NAMES: tuple[str, ...] = (
    "none",
    "marker_follows",
    "press_carrier",
    "zonal_shift",
    "lane_cover",
    "learned",
)

MIN_SEPARATION_M = 0.5
PITCH_MARGIN_M = 2.0
MARKER_RADIUS_M = 10.0  # marker_follows: only a defender this close follows
PRESS_STANDOFF_M = 1.5  # press_carrier: the presser stops this far short of the carrier
ZONAL_LATERAL_FRACTION = 0.5  # zonal_shift: share of the ball's y displacement
ZONAL_LONGITUDINAL_FRACTION = 0.3  # zonal_shift: share of the ball's x displacement
MIN_REPORTED_MOVE_M = 0.01  # smaller differential moves are dropped (numerical noise)


@dataclass
class Movement:
    player_id: str
    from_xy: tuple[float, float]
    to_xy: tuple[float, float]
    provenance: Provenance
    reason: str


@dataclass
class Intervention:
    player_id: str
    x: float
    y: float


@dataclass
class ResponseContext:
    pitch: PitchSpec = field(default_factory=PitchSpec)
    horizon_s: float = 1.5
    limits: KinematicLimits = DEFAULT_LIMITS
    follow_fraction: float = 0.5
    ball_velocity: F | None = None  # canonical frame, causal; None if unavailable
    model: ResponseModel | None = None


# ----------------------------------------------------------------------------------------
# Policy descriptions (served by the API)

_DIFFERENTIAL = (
    "Differential response: a defender moves by the change in the policy's target position "
    "caused by the intervention, so an unchanged situation produces no move."
)
_KINEMATIC = (
    "Each move is limited to the defender's kinematic envelope over the horizon "
    "({limits}) from the observed position and causal velocity. A reachable move is not "
    "a tactically plausible one."
)
_WHO = (
    "Only outfield defenders are moved; attackers, goalkeepers and user-moved players never "
    "are. Moves respect the pitch bounds and a 0.5 m separation (resolved by shortening the "
    "move, in player-id order)."
)
_NOT_WHAT = "This is not what the defender would have done; no ground truth exists for it."


@dataclass(frozen=True)
class PolicySpec:
    name: str
    description: str
    assumptions: tuple[str, ...]


def _specs(limits: KinematicLimits = DEFAULT_LIMITS) -> dict[str, PolicySpec]:
    kin = _KINEMATIC.format(limits=limits.describe())
    common = (_DIFFERENTIAL, kin, _WHO, _NOT_WHAT)
    return {
        "none": PolicySpec(
            "none",
            "No opponent response: every player who was not moved keeps the observed position.",
            ("Defenders do not react to the intervention.",),
        ),
        "marker_follows": PolicySpec(
            "marker_follows",
            "Rule (legacy): the defender nearest to a relocated attacker's observed position "
            f"(within {MARKER_RADIUS_M:.0f} m) shifts by follow_fraction of the attacker's "
            "displacement.",
            (
                "One marker per relocated attacker; no kinematic limit is applied (the move "
                "is still feasibility-checked like user moves).",
                _NOT_WHAT,
            ),
        ),
        "press_carrier": PolicySpec(
            "press_carrier",
            "Rule: when the carrier (with the ball) is relocated, the nearest outfield "
            f"defender closes toward the new carrier position, stopping {PRESS_STANDOFF_M} m "
            "short, as far as the kinematic envelope allows.",
            (
                "Only one defender presses; teammates do not cover the space it leaves.",
                *common,
            ),
        ),
        "zonal_shift": PolicySpec(
            "zonal_shift",
            "Rule: the outfield defenders shift "
            f"{ZONAL_LATERAL_FRACTION:.0%} of the ball's lateral and "
            f"{ZONAL_LONGITUDINAL_FRACTION:.0%} of its longitudinal displacement on top of "
            "their current motion, each clipped to its kinematic envelope.",
            (
                "The shift fractions are a priori assumptions, not fitted to data.",
                "Triggered only when the ball moves (i.e. the carrier is relocated).",
                *common,
            ),
        ),
        "lane_cover": PolicySpec(
            "lane_cover",
            "Rule: when a non-carrier attacker is relocated, the outfield defender who can "
            "get closest to the new carrier-to-attacker passing lane moves toward the nearest "
            "point of that lane, clipped to its kinematic envelope.",
            (
                "One defender per relocated attacker, chosen by reachability of the lane.",
                "The lane is a straight segment to the attacker's position (no lead pass).",
                *common,
            ),
        ),
        "learned": PolicySpec(
            "learned",
            "Ridge model of short-horizon defender displacement fit on real tracking "
            "(per provider, leave-one-match-out evaluated), applied to every outfield "
            "defender as the difference between its forecast for the hypothetical and the "
            "observed state, clipped to the kinematic envelope.",
            (
                "The model forecasts how defenders moved in real play (observational). Its "
                "validation does not cover responses to hypothetical interventions.",
                "Forecast horizon is capped at the longest trained horizon (1.0 s).",
                "The fitted model is dominated by the defender's own velocity (close to damped "
                "constant velocity), so its responses to interventions are very small "
                "(centimetres; docs/RESEARCH.md). Validated forecasting does not make it a "
                "strong response model.",
                "Linear in causal features: own velocity, ball offset and velocity, "
                "nearest-attacker offset, own-goal offset.",
                *common,
            ),
        ),
    }


POLICIES = _specs()


@dataclass(frozen=True)
class PolicyInfo:
    name: str
    description: str
    assumptions: list[str]
    validated: bool
    validation_ref: str | None
    status: str


def policy_info(name: str, provider: str | None = None) -> PolicyInfo:
    """Description plus validation status. ``provider=None`` means validated on all providers.

    ``validated`` is true only for a policy whose short-horizon forecast error beat constant
    velocity in ``research.response_validation`` for that provider (pre-specified rule:
    95% CI of the paired FDE difference below zero at 0.5 s and 1.0 s).
    """
    spec = POLICIES[name]
    validated, ref = False, None
    if name == "learned":
        providers = [provider] if provider else list(PROVIDER_RESOURCES)
        vals = []
        refs = []
        for p in providers:
            m = load_response_model(p)
            v = (m.meta.get("validation") or {}) if m else {}
            vals.append(bool(v.get("validated", False)))
            if v.get("ref"):
                refs.append(str(v["ref"]))
        validated = bool(vals) and all(vals)
        ref = "; ".join(refs) or None
    if name == "none":
        status = "no response simulated"
    elif validated:
        status = "validated: short-horizon forecasts of real play beat constant velocity"
    else:
        status = "experimental: not validated against tracking"
    return PolicyInfo(name, spec.description, list(spec.assumptions), validated, ref, status)


# ----------------------------------------------------------------------------------------
# Helpers


def _outfield_defenders(state: FrameState, moved: set[str]) -> list[int]:
    return [
        j
        for j, t in enumerate(state.teams)
        if t != state.attacking_team
        and not bool(state.is_gk[j])
        and not np.isnan(state.xy[j]).any()
        and state.player_ids[j] not in moved
    ]


def _nearest_on_segment(a: F, b: F, q: F) -> F:
    ab = b - a
    den = float(ab @ ab)
    if den < 1e-12:
        return a.copy()
    t = float(np.clip((q - a) @ ab / den, 0.0, 1.0))
    return a + t * ab


def _press_target(state: FrameState, j: int, carrier: int, standoff: float) -> F:
    p = state.xy[j]
    c = state.xy[carrier]
    d = float(np.linalg.norm(c - p))
    if d < 1e-9:
        return p.copy()
    return p + max(d - standoff, 0.0) * (c - p) / d


def _min_sep(xy: F, j: int, pos: F) -> float:
    others = np.delete(xy, j, axis=0)
    d = np.linalg.norm(others - pos[None, :], axis=1)
    return float(np.nanmin(d)) if np.isfinite(d).any() else float("inf")


@dataclass
class _Candidate:
    j: int
    delta: F
    requested: F
    env: Envelope
    why: str


def _commit(
    obs: FrameState,
    hyp: FrameState,
    cands: list[_Candidate],
    ctx: ResponseContext,
    policy: str,
) -> tuple[list[Movement], list[str]]:
    """Apply candidate moves in player-id order with pitch and separation constraints."""
    moves: list[Movement] = []
    notes: list[str] = []
    hl = ctx.pitch.half_length + PITCH_MARGIN_M
    hw = ctx.pitch.half_width + PITCH_MARGIN_M
    for c in sorted(cands, key=lambda c: obs.player_ids[c.j]):
        if float(np.linalg.norm(c.delta)) < MIN_REPORTED_MOVE_M:
            continue
        p = obs.xy[c.j].copy()
        new = p + c.delta
        # Bounds never pull in a player who is already observed outside them.
        new = np.array(
            [
                np.clip(new[0], min(-hl, p[0]), max(hl, p[0])),
                np.clip(new[1], min(-hw, p[1]), max(hw, p[1])),
            ]
        )
        extra = ""
        if _min_sep(hyp.xy, c.j, new) < MIN_SEPARATION_M:
            if _min_sep(hyp.xy, c.j, p) < MIN_SEPARATION_M:
                notes.append(
                    f"{policy}: {obs.player_ids[c.j]} not moved (would overlap another player)"
                )
                continue
            lo, hi = 0.0, 1.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if _min_sep(hyp.xy, c.j, p + mid * (new - p)) >= MIN_SEPARATION_M:
                    lo = mid
                else:
                    hi = mid
            new = p + lo * (new - p)
            extra = f"; shortened to {lo:.0%} to keep {MIN_SEPARATION_M} m separation"
            if float(np.linalg.norm(new - p)) < MIN_REPORTED_MOVE_M:
                continue
        hyp.xy[c.j] = new
        hyp.position_provenance[c.j] = Provenance.SIMULATED
        got = float(np.linalg.norm(new - p))
        req = float(np.linalg.norm(c.requested))
        notes_env = "".join(f"; {n}" for n in c.env.notes)
        est = (
            "; starting position extrapolated by the provider (estimated)"
            if obs.position_provenance[c.j] == Provenance.ESTIMATED
            else ""
        )
        moves.append(
            Movement(
                obs.player_ids[c.j],
                (float(p[0]), float(p[1])),
                (float(new[0]), float(new[1])),
                Provenance.SIMULATED,
                f"{policy}: {c.why}; target shift {req:.1f} m, applied {got:.1f} m within "
                f"{c.env.horizon_s:.1f} s ({c.env.limits.describe()}){extra}{notes_env}{est} "
                "(simulated response, not observed)",
            )
        )
    return moves, notes


def _differential(
    obs: FrameState,
    hyp: FrameState,
    responders: list[int],
    target: Callable[[FrameState, int, Envelope], F],
    horizon_s: float,
    ctx: ResponseContext,
    why: Callable[[int], str],
    envs: dict[int, Envelope] | None = None,
) -> list[_Candidate]:
    out = []
    for j in responders:
        env = (envs or {}).get(j) or reachability_envelope(
            obs.xy[j], obs.vel[j], horizon_s, ctx.limits
        )
        g_obs, g_hyp = target(obs, j, env), target(hyp, j, env)
        delta = env.clip(g_hyp) - env.clip(g_obs)
        out.append(_Candidate(j, delta, g_hyp - g_obs, env, why(j)))
    return out


# ----------------------------------------------------------------------------------------
# Policies


def _marker_follows(
    state: FrameState,
    hyp: FrameState,
    interventions: list[Intervention],
    moved_ids: set[str],
    follow_fraction: float,
) -> list[Movement]:
    # Legacy rule, unchanged: when an attacker is relocated, the defender nearest to the
    # attacker's observed position (within 10 m) shifts by a fraction of the same
    # displacement. Relocated defenders trigger no response.
    moves: list[Movement] = []
    shifted: set[str] = set()
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
        if d[j] > MARKER_RADIUS_M:
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
    return moves


def _press_carrier(
    obs: FrameState, hyp: FrameState, moved: set[str], ctx: ResponseContext
) -> tuple[list[_Candidate], list[str]]:
    c = obs.carrier
    if c is None:
        return [], ["press_carrier: no ball carrier identified at this frame; no response"]
    cid = obs.player_ids[c]
    if cid not in moved:
        return [], ["press_carrier: the carrier was not relocated; no response"]
    resp = _outfield_defenders(obs, moved)
    if not resp:
        return [], ["press_carrier: no outfield defender available"]
    d = [float(np.linalg.norm(obs.xy[j] - hyp.xy[c])) for j in resp]
    j = resp[int(np.argmin(d))]
    cands = _differential(
        obs,
        hyp,
        [j],
        lambda s, k, _e: _press_target(s, k, c, PRESS_STANDOFF_M),
        ctx.horizon_s,
        ctx,
        lambda _k: f"nearest outfield defender closes toward the relocated carrier {cid}",
    )
    return cands, []


def _zonal_shift(
    obs: FrameState, hyp: FrameState, moved: set[str], ctx: ResponseContext
) -> tuple[list[_Candidate], list[str]]:
    db = hyp.ball - obs.ball
    if np.isnan(db).any() or float(np.linalg.norm(db)) < 1e-9:
        return [], ["zonal_shift: the ball was not moved; no response"]
    shift = np.array([ZONAL_LONGITUDINAL_FRACTION * db[0], ZONAL_LATERAL_FRACTION * db[1]])

    def target(s: FrameState, _j: int, env: Envelope) -> F:
        base = np.array(env.coast_point)
        return base + shift if s is hyp else base

    cands = _differential(
        obs,
        hyp,
        _outfield_defenders(obs, moved),
        target,
        ctx.horizon_s,
        ctx,
        lambda _k: (
            f"block shifts {ZONAL_LATERAL_FRACTION:.0%} lateral / "
            f"{ZONAL_LONGITUDINAL_FRACTION:.0%} longitudinal of the ball's "
            f"{float(np.linalg.norm(db)):.1f} m displacement"
        ),
    )
    return cands, []


def _lane_cover(
    obs: FrameState,
    hyp: FrameState,
    interventions: list[Intervention],
    moved: set[str],
    ctx: ResponseContext,
) -> tuple[list[_Candidate], list[str]]:
    c = obs.carrier
    if c is None:
        return [], ["lane_cover: no ball carrier identified at this frame; no response"]
    targets = [
        obs.index(iv.player_id)
        for iv in interventions
        if obs.teams[obs.index(iv.player_id)] == obs.attacking_team and obs.index(iv.player_id) != c
    ]
    if not targets:
        return [], ["lane_cover: no non-carrier attacker was relocated; no response"]
    resp = _outfield_defenders(obs, moved)
    envs = {
        j: reachability_envelope(obs.xy[j], obs.vel[j], ctx.horizon_s, ctx.limits) for j in resp
    }
    cands: list[_Candidate] = []
    used: set[int] = set()
    notes: list[str] = []
    for a in targets:
        free = [j for j in resp if j not in used]
        if not free:
            notes.append("lane_cover: no outfield defender left to cover a lane")
            break

        def lane_point(s: FrameState, j: int, env: Envelope, a: int = a) -> F:
            return _nearest_on_segment(s.xy[c], s.xy[a], np.array(env.coast_point))

        def rank(j: int) -> tuple[float, float, int]:
            q = lane_point(hyp, j, envs[j])
            gap = float(np.linalg.norm(envs[j].clip(q) - q))
            return gap, float(np.linalg.norm(q - np.array(envs[j].coast_point))), j

        j = min(free, key=rank)
        used.add(j)
        why = (
            f"defender best placed to reach the lane {obs.player_ids[c]} -> "
            f"{obs.player_ids[a]} covers it"
        )

        def reason(_k: int, w: str = why) -> str:
            return w

        cands += _differential(obs, hyp, [j], lane_point, ctx.horizon_s, ctx, reason, envs)
    return cands, notes


def _learned(
    obs: FrameState, hyp: FrameState, moved: set[str], ctx: ResponseContext
) -> tuple[list[_Candidate], list[str]]:
    model = ctx.model
    if model is None:
        raise ValueError("the learned response model is not available for this provider")
    resp = _outfield_defenders(obs, moved)
    if not resp:
        return [], ["learned: no outfield defender available"]
    k = model.offset_index(ctx.horizon_s)
    h = model.offsets_s[k]
    sign = obs.attack_sign
    hl = ctx.pitch.half_length
    bv = (
        np.full(2, np.nan)
        if ctx.ball_velocity is None
        else np.asarray(ctx.ball_velocity, float) * sign
    )
    idx = np.array(resp)

    def forecast(s: FrameState) -> F:
        att = np.array([i for i, t in enumerate(s.teams) if t == s.attacking_team])
        xx, xy = features(
            s.xy[idx] * sign, s.vel[idx] * sign, s.ball * sign, bv, s.xy[att] * sign, hl
        )
        return np.asarray(s.xy[idx] + model.predict(xx, xy, k) * sign)

    g_obs, g_hyp = forecast(obs), forecast(hyp)
    pos = {j: r for r, j in enumerate(resp)}
    cands = _differential(
        obs,
        hyp,
        resp,
        lambda s, j, _e: (g_hyp if s is hyp else g_obs)[pos[j]],
        h,
        ctx,
        lambda _j: f"learned displacement model ({h:.2f} s forecast) responds to the change",
    )
    return cands, []


def apply_response(
    observed: FrameState,
    hypothetical: FrameState,
    interventions: list[Intervention],
    policy: str,
    ctx: ResponseContext,
) -> tuple[list[Movement], list[str]]:
    """Move defenders in ``hypothetical`` (in place). Returns the moves and policy notes."""
    if policy not in POLICY_NAMES:
        raise ValueError(f"unknown response policy {policy!r}")
    moved = {iv.player_id for iv in interventions}
    if policy == "none":
        return [], []
    if policy == "marker_follows":
        return _marker_follows(
            observed, hypothetical, interventions, moved, ctx.follow_fraction
        ), []
    if policy == "press_carrier":
        cands, notes = _press_carrier(observed, hypothetical, moved, ctx)
    elif policy == "zonal_shift":
        cands, notes = _zonal_shift(observed, hypothetical, moved, ctx)
    elif policy == "lane_cover":
        cands, notes = _lane_cover(observed, hypothetical, interventions, moved, ctx)
    else:
        cands, notes = _learned(observed, hypothetical, moved, ctx)
    moves, more = _commit(observed, hypothetical, cands, ctx, policy)
    if not moves and not notes:
        notes.append(f"{policy}: the intervention changes no defender's target noticeably")
    return moves, notes + more
