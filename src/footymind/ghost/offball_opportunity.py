"""Off-ball opportunity under reactive defenders (session 4, ESTIMATED, model-based).

Question
--------
An attacker ``a`` moved off the ball over the last ``horizon_s`` (1.0 s) before a pass
decision. How much did that movement change the team's *modelled* passing opportunity, once
plausible defender responses to it are accounted for, and how much does the answer depend on
the response assumption? The study design and hypotheses are pre-registered in
``research/offball_study.py`` and ``docs/research/offball_opportunity.md``.

Definition (fixed before any match data was analysed)
------------------------------------------------------
At the analysed frame ``t`` (for pass decisions: the 0.2 s guard frame, ADR-0007) let
``V(S)`` be the passing opportunity of the team in possession under the serving option model,
with three measures:

* ``forward`` (primary): the best completion probability among progressive options
  (teammates more than ``PROGRESSIVE_M`` ahead of the ball; 0 if none), as in
  ``movement_decomposition``;
* ``sum``: summed completion probability over all teammates;
* ``own``: the completion probability of the option to ``a`` itself (``run_value``'s quantity).

The counterfactual ``S_cf(a)`` places ``a`` alone at their observed position and causal
velocity ``horizon_s`` earlier ("had not made the run"). The carrier, the ball and every other
attacker keep their observed state. The modelled opportunity change is::

    change_k(a) = V(S_fact,k) - V(S_cf,k(a))

for four defender assumptions ``k``:

1. ``static``: defenders at their observed positions in both states (the ``run_value``
   convention; the observed positions already contain any reaction to the run).
2. ``constant_velocity``: a world in which no defender reacted to anything during the window.
   In both states every outfield defender is at ``x_j(t - h) + v_j(t - h) * h`` (velocity
   ``v_j(t - h)``), and ``a``'s run is valued against that background. Defenders whose earlier
   position or velocity is unknown keep their observed state (counted). The factual of this
   assumption is therefore not the observed state.
3. ``marker_follows`` and 4. ``lane_cover``: the defenders are as observed *minus* the response
   the policy attributes to ``a``'s movement. The policy is run with ``a`` moved from ``x_a(t)``
   back by the attributed movement ``u_a``, through ``counterfactual.response.apply_response``
   (differential, kinematically clipped for ``lane_cover``, horizon ``horizon_s``), and the
   simulated defender moves are applied to ``S_cf(a)``.

The attributed movement ``u_a`` is, by default, the **innovation**: the displacement minus the
attacker's own constant-velocity forecast, ``x_a(t) - x_a(t - h) - v_a(t - h) * h``. Claim C37
found that these two rules reproduce real defender reactions to attackers' *unexpected* movement
(beta about 1.0-1.3) and that the full displacement double counts motion the defenders were
already making (beta 0.02-0.47). So the innovation attributes to ``a`` only the response C37
supports; it is conservative. ``attribution="full"`` (the whole displacement) is a sensitivity
variant and an upper bound. If ``v_a(t - h)`` is unknown the innovation is unknown, and the
policy changes are NaN (never a fabricated 0).

What this is not: a causal effect, a player value or a prediction of what defenders would
have done. The mixed states never happened, the option model extrapolates to them, and the
rule responses are assumptions that passed only a necessary observational check (C37). All
outputs are ``Provenance.ESTIMATED``; moved defenders are ``SIMULATED`` and moved attackers
``HYPOTHETICAL`` in the counterfactual states.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from footymind.counterfactual.response import Intervention, ResponseContext, apply_response
from footymind.spatial.analysis import pass_options
from footymind.spatial.pitch_control import DEFAULT_PARAMS, PitchControlParams
from footymind.spatial.state import FrameState
from footymind.tracking.model import PitchSpec, Provenance

F = npt.NDArray[np.float64]

PROGRESSIVE_M = 5.0
MIN_MOVE_M = 0.01  # a response smaller than this is not counted as a move
HORIZON_S = 1.0
MEASURES = ("forward", "sum", "own")
ASSUMPTIONS = ("static", "constant_velocity", "marker_follows", "lane_cover")
POLICY_ASSUMPTIONS = ("marker_follows", "lane_cover")
ATTRIBUTIONS = ("innovation", "full")

ASSUMPTION_TEXT = {
    "static": "Defenders stay where they were observed: the observed positions already contain "
    "any reaction to the run, so the run is credited against defenders who reacted.",
    "constant_velocity": "No defender reacted to anything in the last second: every outfield "
    "defender continues their motion of one second earlier, and the run is valued against "
    "that background.",
    "marker_follows": "The defender nearest the attacker (within 10 m) followed half of the "
    "attacker's unexpected movement; without the run that response is removed. Rule-based; "
    "passed an observational check (C37), not validated.",
    "lane_cover": "The defender best placed to cover the carrier-to-attacker lane moved toward "
    "it in response to the attacker's unexpected movement, within kinematic limits; without "
    "the run that response is removed. Rule-based; passed an observational check (C37), not "
    "validated.",
}


@dataclass(frozen=True)
class AttackerOpportunity:
    player_id: str
    displacement_m: float
    innovation_m: float  # NaN if the attacker's earlier velocity is unknown
    # assumption -> measure -> V(factual) - V(counterfactual); NaN if not defined
    change: dict[str, dict[str, float]]
    # assumption -> number of defenders moved by the attributed response (policies only)
    defenders_moved: dict[str, int]
    provenance: Provenance = Provenance.ESTIMATED


@dataclass(frozen=True)
class OffballOpportunity:
    attribution: str
    horizon_s: float
    factual: dict[str, dict[str, float]]  # "observed" / "constant_velocity" -> measure -> V
    attackers: list[AttackerOpportunity]
    n_cv_unresolved_defenders: int  # outfield defenders kept at t (unknown earlier state)
    notes: list[str] = field(default_factory=list)
    provenance: Provenance = Provenance.ESTIMATED


def opportunity(st: FrameState, params: PitchControlParams = DEFAULT_PARAMS) -> dict[str, float]:
    """``forward`` and ``sum`` for the team, plus ``own:<player_id>`` for every option."""
    opts = pass_options(st, params)
    vals = [o.p_success_model if o.p_success_model is not None else o.p_success for o in opts]
    fwd = [v for v, o in zip(vals, opts, strict=True) if o.forward_progress_m > PROGRESSIVE_M]
    out = {
        "forward": float(max(fwd)) if fwd else 0.0,
        "sum": float(sum(vals)) if vals else np.nan,
    }
    for v, o in zip(vals, opts, strict=True):
        out["own:" + o.receiver_id] = float(v)
    return out


def _measures(v: dict[str, float], pid: str) -> dict[str, float]:
    return {"forward": v["forward"], "sum": v["sum"], "own": v.get("own:" + pid, np.nan)}


def _diff(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {m: a[m] - b[m] for m in MEASURES}


NAN_CHANGE = {m: np.nan for m in MEASURES}


def eligible_attackers(st: FrameState, earlier_xy: F) -> list[int]:
    """Off-ball outfield attackers on the pitch with an earlier position (carrier excluded)."""
    assert st.attacking_team is not None
    on = st.on_pitch() & ~np.isnan(earlier_xy).any(axis=1)
    att = st.team_mask(st.attacking_team) & on & ~st.is_gk
    if st.carrier is not None:
        att[st.carrier] = False
    return [int(i) for i in np.flatnonzero(att)]


def cv_background(
    st: FrameState, earlier_xy: F, earlier_vel: F, horizon_s: float
) -> tuple[FrameState, int]:
    """The observed state with outfield defenders at their constant-velocity extrapolation."""
    s = st.copy()
    unresolved = 0
    assert st.attacking_team is not None
    dfn = (~st.team_mask(st.attacking_team)) & st.on_pitch() & ~st.is_gk
    for j in np.flatnonzero(dfn):
        if np.isnan(earlier_xy[j]).any() or np.isnan(earlier_vel[j]).any():
            unresolved += 1
            continue
        s.xy[j] = earlier_xy[j] + earlier_vel[j] * horizon_s
        s.vel[j] = earlier_vel[j]
        s.position_provenance[j] = Provenance.HYPOTHETICAL
    return s, unresolved


def offball_opportunity(
    st: FrameState,
    earlier_xy: F,
    earlier_vel: F,
    horizon_s: float = HORIZON_S,
    attribution: str = "innovation",
    pitch: PitchSpec | None = None,
    attackers: list[int] | None = None,
    assumptions: tuple[str, ...] = ASSUMPTIONS,
    params: PitchControlParams = DEFAULT_PARAMS,
) -> OffballOpportunity:
    """Per-attacker modelled opportunity change under each defender assumption.

    ``st`` is the observed state at ``t``; ``earlier_xy`` / ``earlier_vel`` are every player's
    observed position and causal velocity ``horizon_s`` earlier (NaN if unknown). Pure: reads no
    data beyond its arguments, so causality is the caller's responsibility (pass the guard frame).
    """
    if st.attacking_team is None or st.carrier is None or np.isnan(st.ball).any():
        raise ValueError("needs a possession, a carrier and an observed ball")
    if attribution not in ATTRIBUTIONS:
        raise ValueError(f"unknown attribution {attribution!r}")
    unknown = [k for k in assumptions if k not in ASSUMPTIONS]
    if unknown:
        raise ValueError(f"unknown assumptions {unknown}")
    ctx = ResponseContext(pitch=pitch or PitchSpec(), horizon_s=horizon_s)
    v_obs = opportunity(st, params)
    cv_state, n_unres = (None, 0)
    v_cv: dict[str, float] | None = None
    if "constant_velocity" in assumptions:
        cv_state, n_unres = cv_background(st, earlier_xy, earlier_vel, horizon_s)
        v_cv = opportunity(cv_state, params)
    idx = attackers if attackers is not None else eligible_attackers(st, earlier_xy)
    out: list[AttackerOpportunity] = []
    for a in idx:
        pid = st.player_ids[a]
        disp = st.xy[a] - earlier_xy[a]
        innov = (
            disp - earlier_vel[a] * horizon_s
            if not np.isnan(earlier_vel[a]).any()
            else np.full(2, np.nan)
        )
        u = innov if attribution == "innovation" else disp

        def cf_from(base: FrameState, a: int = a) -> FrameState:
            s = base.copy()
            s.xy[a] = earlier_xy[a]
            s.vel[a] = earlier_vel[a]
            s.position_provenance[a] = Provenance.HYPOTHETICAL
            return s

        change: dict[str, dict[str, float]] = {}
        moved: dict[str, int] = {}
        for k in assumptions:
            if k == "static":
                change[k] = _diff(
                    _measures(v_obs, pid), _measures(opportunity(cf_from(st), params), pid)
                )
            elif k == "constant_velocity":
                assert cv_state is not None and v_cv is not None
                change[k] = _diff(
                    _measures(v_cv, pid), _measures(opportunity(cf_from(cv_state), params), pid)
                )
            else:
                if np.isnan(u).any():
                    change[k], moved[k] = dict(NAN_CHANGE), 0
                    continue
                hyp = st.copy()
                hyp.xy[a] = st.xy[a] - u
                hyp.position_provenance[a] = Provenance.HYPOTHETICAL
                iv = Intervention(pid, float(hyp.xy[a, 0]), float(hyp.xy[a, 1]))
                moves, _ = apply_response(st, hyp, [iv], k, ctx)
                cf = cf_from(st)
                n_moved = 0
                for m in moves:
                    j = st.index(m.player_id)
                    cf.xy[j] = m.to_xy
                    cf.position_provenance[j] = Provenance.SIMULATED
                    n_moved += int(np.hypot(*np.subtract(m.to_xy, m.from_xy)) >= MIN_MOVE_M)
                moved[k] = n_moved
                change[k] = _diff(_measures(v_obs, pid), _measures(opportunity(cf, params), pid))
        out.append(
            AttackerOpportunity(
                player_id=pid,
                displacement_m=float(np.linalg.norm(disp)),
                innovation_m=float(np.linalg.norm(innov)),
                change=change,
                defenders_moved=moved,
            )
        )
    factual = {"observed": {m: v_obs[m] for m in ("forward", "sum")}}
    if v_cv is not None:
        factual["constant_velocity"] = {m: v_cv[m] for m in ("forward", "sum")}
    return OffballOpportunity(
        attribution=attribution,
        horizon_s=horizon_s,
        factual=factual,
        attackers=out,
        n_cv_unresolved_defenders=n_unres,
    )


def rotate_history(
    st: FrameState, earlier_xy: F, earlier_vel: F, idx: list[int], rng: np.random.Generator
) -> tuple[F, F]:
    """Placebo: each listed attacker keeps the size of their displacement over the window, but
    its direction (and the earlier velocity with it) is rotated by a random angle. The
    innovation rotates with them, so its size is kept too. Defenders are untouched.
    """
    xy, vel = earlier_xy.copy(), earlier_vel.copy()
    for a in idx:
        ang = rng.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        xy[a] = st.xy[a] - rot @ (st.xy[a] - earlier_xy[a])
        if not np.isnan(vel[a]).any():
            vel[a] = rot @ vel[a]
    return xy, vel
