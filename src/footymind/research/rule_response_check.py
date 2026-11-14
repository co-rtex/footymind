"""Observational check of the rule-based response policies (pre-registered, session 3).

``python -m footymind.research.rule_response_check <provider> [n_per_match]`` writes
``experiments/results/rule_response_check_v1[_<provider>]/results.json``.

No ground truth exists for responses to *hypothetical* interventions. Real play still
contains natural "interventions": the attackers really do move. This check treats the
attackers' real movement over the next ``HORIZON_S`` as the intervention. At a pass start
frame ``t``:

* ``press_carrier`` / ``zonal_shift``: the carrier (and the ball with it) is moved to the
  carrier's real position at ``t + h``;
* ``lane_cover`` / ``marker_follows``: every non-carrier attacker is moved to their real
  position at ``t + h``.

The policy's simulated displacement of each defender it moves is added to that defender's
constant-velocity forecast, and both forecasts are compared with where the defender really
was at ``t + h`` (detected positions only on SkillCorner). The rule counts as
**observationally supported** if this lowers the final displacement error of the defenders
it moves (possession-clustered 95% CI below 0). That is a necessary condition for a
plausible response, not evidence about responses to interventions that never happened.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
import polars as pl

from footymind.counterfactual.kinematics import DEFAULT_LIMITS, Limits, default_limits
from footymind.counterfactual.response import Intervention, ResponseContext, apply_response
from footymind.data.store import available_matches, git_revision, load_match
from footymind.paths import EXPERIMENTS_DIR
from footymind.research.pass_dataset import DatasetConfig, _prev_event_features, label_events
from footymind.research.response_validation import ball_velocities
from footymind.spatial.state import frame_state

HORIZON_S = 1.0
GUARD_S = 0.2  # v2: evaluate from the guard frame (ADR-0007)
# v2 intervention designs (added after an independent review, 2026-11-14):
#   full        the attackers' full real displacement (the v1 design). The defenders'
#               constant-velocity forecast already contains their co-movement, so this can
#               double count.
#   innovation  only the part of each attacker's displacement that their own constant
#               velocity did not predict (real minus constant-velocity position).
#   placebo     the innovations permuted among the moved attackers of the same frame. This
#               keeps team-wide movement and breaks the attacker-defender pairing.
DESIGNS = ("full", "innovation", "placebo")
CARRIER_POLICIES = ("press_carrier", "zonal_shift")
ATTACKER_POLICIES = ("lane_cover", "marker_follows")


def _rows(
    provider: str,
    n_per_match: int,
    seed: int,
    guard_s: float = 0.0,
    designs: tuple[str, ...] = ("full",),
    limits: Limits = DEFAULT_LIMITS,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for mid in [m for m in available_matches() if m.startswith(provider + "-")]:
        b = load_match(mid)
        td = b.tracking
        bv = ball_velocities(mid)
        k = round(HORIZON_S * td.fps)
        ev = label_events(_prev_event_features(b.events), DatasetConfig())
        ev = ev.sample(min(n_per_match, ev.height), seed=seed)
        for r in ev.iter_rows(named=True):
            i = td.index_of_frame(int(r["start_frame"])) - round(guard_s * td.fps)
            if i < 0 or i + k >= td.n_frames or td.period[i + k] != td.period[i]:
                continue
            f = int(td.frame[i])
            obs = frame_state(mid, f)
            if obs.attacking_team is None or obs.carrier is None or np.isnan(obs.ball).any():
                continue
            future = td.xy[i + k].astype(np.float64)
            det = td.detected[i + k] if td.detected is not None else None
            det_now = td.detected[i] if td.detected is not None else None
            att = np.flatnonzero(obs.team_mask(obs.attacking_team) & obs.on_pitch())
            ctx = ResponseContext(
                pitch=td.pitch, horizon_s=HORIZON_S, ball_velocity=bv[i], limits=limits
            )
            for design in designs:
                for policy in (*CARRIER_POLICIES, *ATTACKER_POLICIES):
                    movers = (
                        [obs.carrier]
                        if policy in CARRIER_POLICIES
                        else [a for a in att if a != obs.carrier]
                    )
                    movers = [int(a) for a in movers if not np.isnan(future[a]).any()]
                    if not movers:
                        continue
                    disp = {}
                    for a in movers:
                        full = future[a] - obs.xy[a]
                        cv_a = np.nan_to_num(obs.vel[a]) * HORIZON_S
                        disp[a] = full if design == "full" else full - cv_a
                    if design == "placebo":
                        pool = [
                            int(a) for a in att if a != obs.carrier and int(a) in disp
                        ] or movers
                        if len(movers) == 1:  # carrier policies: borrow another attacker's
                            others = [
                                a for a in att if a != obs.carrier and not np.isnan(future[a]).any()
                            ]
                            if not others:
                                continue
                            donor = int(rng.choice(others))
                            vb = np.nan_to_num(obs.vel[donor]) * HORIZON_S
                            disp = {movers[0]: future[donor] - obs.xy[donor] - vb}
                        else:
                            perm = rng.permutation(len(pool))
                            disp = {a: disp[pool[perm[n]]] for n, a in enumerate(pool)}
                    ivs = [
                        Intervention(
                            obs.player_ids[a],
                            float(obs.xy[a, 0] + disp[a][0]),
                            float(obs.xy[a, 1] + disp[a][1]),
                        )
                        for a in movers
                        if a in disp
                    ]
                    if not ivs:
                        continue
                    hyp = obs.copy()
                    for iv in ivs:
                        j = obs.index(iv.player_id)
                        if j == obs.carrier:
                            hyp.ball = hyp.ball + (np.array([iv.x, iv.y]) - obs.xy[j])
                        hyp.xy[j] = (iv.x, iv.y)
                    moves, _ = apply_response(obs, hyp, ivs, policy, ctx)
                    for m in moves:
                        j = obs.index(m.player_id)
                        if np.isnan(future[j]).any():
                            continue
                        if det is not None and det_now is not None and not (det[j] and det_now[j]):
                            continue
                        v = np.nan_to_num(obs.vel[j])
                        cv = obs.xy[j] + v * HORIZON_S
                        delta = np.array(m.to_xy) - np.array(m.from_xy)
                        resid = future[j] - cv
                        out.append(
                            {
                                "match_id": mid,
                                "possession_id": r["possession_id"],
                                "design": design,
                                "policy": policy,
                                "move_m": float(np.linalg.norm(delta)),
                                "err_cv": float(np.linalg.norm(resid)),
                                "err_policy": float(np.linalg.norm(cv + delta - future[j])),
                                # projection of the real residual on the simulated move
                                "proj": float(resid @ delta),
                                "move_sq": float(delta @ delta),
                            }
                        )
    return out


def _summary(df: pl.DataFrame, n_boot: int, seed: int) -> dict[str, Any]:
    d = (df["err_policy"] - df["err_cv"]).to_numpy()
    groups = (df["match_id"] + "|" + df["possession_id"].fill_null("none")).to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(uniq.size)]
    rng = np.random.default_rng(seed)
    boots = [
        float(d[np.concatenate([members[g] for g in rng.integers(0, uniq.size, uniq.size)])].mean())
        for _ in range(n_boot)
    ]
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return {
        "n_moved_defenders": int(df.height),
        "mean_move_m": float(df["move_m"].mean()),  # type: ignore[arg-type]
        "fde_cv_m": float(df["err_cv"].mean()),  # type: ignore[arg-type]
        "fde_policy_m": float(df["err_policy"].mean()),  # type: ignore[arg-type]
        "delta_m": float(d.mean()),
        "ci95": ci,
        "observationally_supported": bool(ci[1] < 0),
        # least-squares slope of the real residual on the simulated move (1 = right size)
        "beta": float(df["proj"].sum()) / max(float(df["move_sq"].sum()), 1e-12)
        if "proj" in df.columns
        else None,
    }


def main(
    provider: str = "metrica",
    n_per_match: int = 150,
    seed: int = 0,
    n_boot: int = 1000,
    version: int = 2,
) -> dict[str, Any]:
    """v1: full design at the release frame (the original check). v2: all designs at the guard.

    v1 and v2 clip with the session-3 envelope (``DEFAULT_LIMITS``) so they reproduce; v3 is
    v2 with the envelope in force for the provider (envelope v2, session 4).
    """
    t0 = time.time()
    guard, designs = (0.0, ("full",)) if version == 1 else (GUARD_S, DESIGNS)
    limits = default_limits(provider) if version >= 3 else DEFAULT_LIMITS
    df = pl.DataFrame(_rows(provider, n_per_match, seed, guard, designs, limits))
    by_design: dict[str, Any] = {}
    for d in designs:
        sub = df.filter(pl.col("design") == d)
        by_design[d] = {
            p: _summary(sub.filter(pl.col("policy") == p), n_boot, seed)
            for p in (*CARRIER_POLICIES, *ATTACKER_POLICIES)
            if (sub["policy"] == p).sum() > 20
        }
    res: dict[str, Any] = {
        "name": f"rule_response_check_v{version}"
        + ("" if provider == "metrica" else f"_{provider}"),
        "provider": provider,
        "code_revision": git_revision(),
        "horizon_s": HORIZON_S,
        "guard_s": guard,
        "envelope": limits.describe(),
        "n_per_match": n_per_match,
        "policies": by_design["full"],
        "designs": by_design,
    }
    res["runtime_s"] = round(time.time() - t0, 1)
    out = EXPERIMENTS_DIR / "results" / res["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


if __name__ == "__main__":
    r = main(
        sys.argv[1] if len(sys.argv) > 1 else "metrica",
        int(sys.argv[2]) if len(sys.argv) > 2 else 150,
        version=int(sys.argv[3]) if len(sys.argv) > 3 else 2,
    )
    print(json.dumps(r["policies"], indent=2))
