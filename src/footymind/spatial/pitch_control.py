"""Physics-based pitch control (time-to-intercept model).

Implements the potential pitch control field (PPCF) model of

    W. Spearman, "Beyond Expected Goals", MIT Sloan Sports Analytics
    Conference, 2018,

and its pass-probability predecessor (Spearman, Basye, Dick, Hotovy & Pop,
"Physics-Based Modeling of Pass Probabilities in Soccer", MIT SSAC 2017).
Default parameters follow the widely used Friends-of-Tracking reference
parameterisation (L. Shaw, 2020). This module is an independent vectorised
re-implementation of the published equations, not a copy of that code.

Epistemic status: the output is an **estimate** (``Provenance.ESTIMATED``)
under literature parameters. It is **not calibrated on the data used here**.
``footymind.research`` evaluates how well it predicts observed pass outcomes.

Model summary
-------------
For a target location ``r`` and player ``j`` at position ``p_j`` with velocity
``v_j``:

* time to intercept: ``tau_j(r) = t_react + |r - (p_j + v_j t_react)| / v_max``.
  The player keeps their velocity for the reaction time, then runs straight
  at maximum speed.
* the probability that ``j`` has arrived by time ``T`` is logistic:
  ``P_j(T) = 1 / (1 + exp(-pi / (sqrt(3) sigma) (T - tau_j)))``.
* the ball arrives at ``t_ball = |r - r_ball| / v_ball``. From then on, control
  accumulates as ``dPPCF_j/dT = (1 - sum_k PPCF_k) P_j(T) lambda_j``.

Team control at ``r`` is the sum over that team's players once the total
converges to 1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import numpy.typing as npt

F = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PitchControlParams:
    max_player_speed: float = 5.0  # m/s, sustained running speed after reaction
    reaction_time: float = 0.7  # s
    tti_sigma: float = 0.45  # s, arrival-time uncertainty
    lambda_att: float = 4.3  # 1/s, ball-control rate
    kappa_def: float = 1.0  # defensive control advantage multiplier
    lambda_gk_multiplier: float = 3.0  # goalkeepers can handle the ball
    average_ball_speed: float = 15.0  # m/s
    int_dt: float = 0.04  # s
    max_int_time: float = 10.0  # s
    converge_tol: float = 0.01

    @property
    def lambda_def(self) -> float:
        return self.lambda_att * self.kappa_def

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def with_(self, **kw: float) -> PitchControlParams:
        return replace(self, **kw)


DEFAULT_PARAMS = PitchControlParams()

# Parameter ranges for sensitivity analysis. They are plausible ranges around
# the literature defaults, not posterior distributions.
SENSITIVITY_RANGES: dict[str, tuple[float, float]] = {
    "max_player_speed": (4.5, 6.0),
    "reaction_time": (0.5, 0.9),
    "tti_sigma": (0.35, 0.55),
    "average_ball_speed": (12.0, 18.0),
}


def time_to_intercept(targets: F, pos: F, vel: F, params: PitchControlParams = DEFAULT_PARAMS) -> F:
    """``(G, N)`` time for each of ``N`` players to reach each of ``G`` targets."""
    reaction_pos = pos + np.nan_to_num(vel) * params.reaction_time  # unknown velocity -> 0
    d = np.linalg.norm(targets[:, None, :] - reaction_pos[None, :, :], axis=-1)
    return params.reaction_time + d / params.max_player_speed


def pitch_control(
    targets: F,
    ball: F,
    att_pos: F,
    att_vel: F,
    def_pos: F,
    def_vel: F,
    att_gk: npt.NDArray[np.bool_] | None = None,
    def_gk: npt.NDArray[np.bool_] | None = None,
    params: PitchControlParams = DEFAULT_PARAMS,
) -> tuple[F, F, F]:
    """Pitch control at ``targets``.

    Returns ``(ppcf_att, ppcf_att_players, ppcf_def_players)`` with shapes
    ``(G,)``, ``(G, A)`` and ``(G, D)``. Players with NaN positions (not on the
    pitch or unobserved) are excluded. Velocities that are NaN are treated as
    zero, and the caller is responsible for flagging that assumption.
    """
    targets = np.atleast_2d(np.asarray(targets, dtype=np.float64))
    g = targets.shape[0]
    att_ok = ~np.isnan(att_pos[:, 0])
    def_ok = ~np.isnan(def_pos[:, 0])
    ap, av = att_pos[att_ok], att_vel[att_ok]
    dp, dv = def_pos[def_ok], def_vel[def_ok]
    a_n, d_n = ap.shape[0], dp.shape[0]
    lam_a = np.full(a_n, params.lambda_att)
    lam_d = np.full(d_n, params.lambda_def)
    if att_gk is not None:
        lam_a[np.asarray(att_gk)[att_ok]] *= params.lambda_gk_multiplier
    if def_gk is not None:
        lam_d[np.asarray(def_gk)[def_ok]] *= params.lambda_gk_multiplier

    out_att = np.zeros((g, a_n))
    out_def = np.zeros((g, d_n))
    if a_n == 0 or d_n == 0 or np.isnan(ball).any():
        full = np.full(g, np.nan)
        return full, np.full((g, att_pos.shape[0]), np.nan), np.full((g, def_pos.shape[0]), np.nan)

    t_ball = np.linalg.norm(targets - ball[None, :], axis=1) / params.average_ball_speed
    tti_a = time_to_intercept(targets, ap, av, params)
    tti_d = time_to_intercept(targets, dp, dv, params)
    tau_a = tti_a.min(axis=1)
    tau_d = tti_d.min(axis=1)

    # Shortcut for locations where one team arrives decisively first, with the
    # same veto construction as the reference parameterisation (3x the time
    # needed to reach ~90% control).
    s = np.sqrt(3) * params.tti_sigma / np.pi
    ttc_att = 3.0 * np.log(10) * (s + 1.0 / params.lambda_att)
    ttc_def = 3.0 * np.log(10) * (s + 1.0 / params.lambda_def)
    def_wins = tau_a - np.maximum(t_ball, tau_d) >= ttc_def
    att_wins = tau_d - np.maximum(t_ball, tau_a) >= ttc_att
    ppcf = np.zeros(g)
    ppcf[att_wins] = 1.0
    # Credit decisive locations to the fastest player
    if att_wins.any():
        out_att[att_wins, np.argmin(tti_a[att_wins], axis=1)] = 1.0
    if def_wins.any():
        out_def[def_wins, np.argmin(tti_d[def_wins], axis=1)] = 1.0

    todo = ~(att_wins | def_wins)
    if todo.any():
        ta, td_ = tti_a[todo], tti_d[todo]
        tb = t_ball[todo]
        # players far slower than the fastest teammate cannot contribute
        incl_a = ta - tau_a[todo, None] < ttc_att
        incl_d = td_ - tau_d[todo, None] < ttc_def
        acc_a = np.zeros_like(ta)
        acc_d = np.zeros_like(td_)
        k_scale = np.pi / (np.sqrt(3.0) * params.tti_sigma)
        n_steps = int(np.ceil(params.max_int_time / params.int_dt))
        active = np.ones(ta.shape[0], dtype=bool)
        for k in range(1, n_steps + 1):
            t = tb[:, None] - params.int_dt + k * params.int_dt
            remaining = 1.0 - acc_a.sum(axis=1) - acc_d.sum(axis=1)
            remaining = np.clip(remaining, 0.0, 1.0)[:, None] * active[:, None]
            pa = 1.0 / (1.0 + np.exp(-k_scale * (t - ta))) * incl_a
            pd = 1.0 / (1.0 + np.exp(-k_scale * (t - td_))) * incl_d
            da = remaining * pa * lam_a[None, :] * params.int_dt
            dd = remaining * pd * lam_d[None, :] * params.int_dt
            acc_a += np.maximum(da, 0.0)
            acc_d += np.maximum(dd, 0.0)
            tot = acc_a.sum(axis=1) + acc_d.sum(axis=1)
            active = tot < 1.0 - params.converge_tol
            if not active.any():
                break
        tot = acc_a.sum(axis=1) + acc_d.sum(axis=1)
        tot = np.where(tot > 0, tot, 1.0)
        out_att[todo] = acc_a / tot[:, None]  # normalise residual non-convergence
        out_def[todo] = acc_d / tot[:, None]
        ppcf[todo] = out_att[todo].sum(axis=1)

    full_att = np.zeros((g, att_pos.shape[0]))
    full_att[:, att_ok] = out_att
    full_def = np.zeros((g, def_pos.shape[0]))
    full_def[:, def_ok] = out_def
    return ppcf, full_att, full_def


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    length: float
    width: float

    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def dy(self) -> float:
        return self.width / self.ny

    def centres(self) -> F:
        xs = -self.length / 2 + self.dx * (np.arange(self.nx) + 0.5)
        ys = -self.width / 2 + self.dy * (np.arange(self.ny) + 0.5)
        gx, gy = np.meshgrid(xs, ys)  # (ny, nx), row-major from y = -W/2 upwards
        return np.stack([gx.ravel(), gy.ravel()], axis=1)


DEFAULT_GRID = Grid(nx=50, ny=32, length=105.0, width=68.0)
