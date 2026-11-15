"""Fit PPCF/lane parameters to observed pass outcomes (leave-one-match-out).

Question: is the literature physics estimate poorly calibrated because of its
*parameters* (fixable by fitting) or its *structure* (static target, inferred receiver,
uniform reaction time)?

Procedure (fixed in advance):
* Use the attempted-pass dataset (same label rules and receiver inference as RQ1).
* Objective: Bernoulli log loss of ``clip(PPCF(target) * p_lane_clear, eps, 1 - eps)``.
* Free parameters and bounds: reaction_time [0.2, 1.2], tti_sigma [0.15, 1.0],
  max_player_speed [3, 8], average_ball_speed [8, 25], lambda_att [1, 10],
  kappa_def [0.5, 2.5].
* Optimiser: 40 Latin-hypercube-style random starts evaluated on a fixed random subsample
  of 800 training passes, then Nelder-Mead from the best start (≤ 120 evaluations) on the
  same subsample. Deterministic seed.
* Evaluate the fitted parameters on all passes of the held-out match. Compare with the
  literature parameters on the same passes.

Usage: ``python -m footymind.research.calibrate_physics`` (writes
``experiments/results/physics_calibration_v1/results.json``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from footymind.data.store import available_matches, git_revision, load_match
from footymind.paths import EXPERIMENTS_DIR
from footymind.research.evaluate import ece
from footymind.research.pass_dataset import DatasetConfig, build
from footymind.spatial.lanes import lane
from footymind.spatial.pitch_control import DEFAULT_PARAMS, PitchControlParams, pitch_control
from footymind.spatial.state import frame_state

EPS = 1e-4
NAMES = [
    "reaction_time",
    "tti_sigma",
    "max_player_speed",
    "average_ball_speed",
    "lambda_att",
    "kappa_def",
]
LO = np.array([0.2, 0.15, 3.0, 8.0, 1.0, 0.5])
HI = np.array([1.2, 1.0, 8.0, 25.0, 10.0, 2.5])


@dataclass
class PassState:
    match_id: str
    y: int
    ball: np.ndarray
    target: np.ndarray
    att: np.ndarray
    att_v: np.ndarray
    dfn: np.ndarray
    dfn_v: np.ndarray
    att_gk: np.ndarray
    def_gk: np.ndarray


def extract(match_ids: list[str]) -> list[PassState]:
    df = build(match_ids, DatasetConfig())
    out = []
    for r in df.iter_rows(named=True):
        s = frame_state(r["match_id"], r["frame"])
        td = load_match(r["match_id"]).tracking
        ball = s.ball
        if np.isnan(ball).any():
            continue
        att = s.team_mask(r["team"]) & s.on_pitch()
        dfn = (~s.team_mask(r["team"])) & s.on_pitch()
        cand = td.player_index(r["inferred_receiver_id"])
        out.append(
            PassState(
                r["match_id"],
                int(r["success"]),
                ball.copy(),
                s.xy[cand].copy(),
                s.xy[att].copy(),
                s.vel[att].copy(),
                s.xy[dfn].copy(),
                s.vel[dfn].copy(),
                s.is_gk[att].copy(),
                s.is_gk[dfn].copy(),
            )
        )
    return out


def predict(states: list[PassState], p: PitchControlParams) -> np.ndarray:
    out = np.empty(len(states))
    for i, s in enumerate(states):
        ppcf, _, _ = pitch_control(
            s.target[None, :], s.ball, s.att, s.att_v, s.dfn, s.dfn_v, s.att_gk, s.def_gk, p
        )
        out[i] = ppcf[0] * lane(s.ball, s.target, s.dfn, s.dfn_v, p).p_lane_clear
    return np.clip(np.nan_to_num(out, nan=0.5), EPS, 1 - EPS)


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def to_params(z: np.ndarray) -> PitchControlParams:
    v = LO + (HI - LO) / (1 + np.exp(-z))  # unconstrained -> bounded
    return DEFAULT_PARAMS.with_(**dict(zip(NAMES, map(float, v), strict=True)))


def fit(train: list[PassState], seed: int = 0, n_sub: int = 800) -> tuple[PitchControlParams, dict]:
    rng = np.random.default_rng(seed)
    sub = [train[i] for i in rng.choice(len(train), size=min(n_sub, len(train)), replace=False)]
    y = np.array([s.y for s in sub])
    n_evals = 0

    def obj(z: np.ndarray) -> float:
        nonlocal n_evals
        n_evals += 1
        return logloss(y, predict(sub, to_params(z)))

    starts = rng.normal(0.0, 1.5, size=(40, len(NAMES)))
    scores = [obj(z) for z in starts]
    z0 = starts[int(np.argmin(scores))]
    res = minimize(obj, z0, method="Nelder-Mead", options={"maxfev": 120, "xatol": 1e-3})
    return to_params(res.x), {"n_evals": n_evals, "subsample_logloss": float(res.fun)}


def main() -> None:
    t0 = time.time()
    matches = [m for m in available_matches() if m.startswith("metrica-sample-")]
    states = extract(matches)
    folds = {}
    all_y, all_lit, all_fit = [], [], []
    for test_m in matches:
        tr = [s for s in states if s.match_id != test_m]
        te = [s for s in states if s.match_id == test_m]
        params, info = fit(tr)
        y = np.array([s.y for s in te])
        p_lit, p_fit = predict(te, DEFAULT_PARAMS), predict(te, params)
        all_y.append(y)
        all_lit.append(p_lit)
        all_fit.append(p_fit)
        folds[test_m] = {
            "n_test": int(y.size),
            "fitted_params": {k: round(getattr(params, k), 4) for k in NAMES},
            "fit_info": info,
            "literature": {"log_loss": logloss(y, p_lit), "ece": ece(y, p_lit)},
            "fitted": {"log_loss": logloss(y, p_fit), "ece": ece(y, p_fit)},
        }
        print(test_m, json.dumps(folds[test_m]))
    y, pl_, pf = map(np.concatenate, (all_y, all_lit, all_fit))
    from sklearn.metrics import roc_auc_score

    result = {
        "name": "physics_calibration_v1",
        "code_revision": git_revision(),
        "matches": matches,
        "n": int(y.size),
        "runtime_s": round(time.time() - t0, 1),
        "pooled": {
            "literature": {
                "log_loss": logloss(y, pl_),
                "ece": ece(y, pl_),
                "auc": float(roc_auc_score(y, pl_)),
            },
            "fitted": {
                "log_loss": logloss(y, pf),
                "ece": ece(y, pf),
                "auc": float(roc_auc_score(y, pf)),
            },
        },
        "folds": folds,
        "bounds": {k: [float(a), float(b)] for k, a, b in zip(NAMES, LO, HI, strict=True)},
    }
    out = EXPERIMENTS_DIR / "results" / "physics_calibration_v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["pooled"], indent=2))


if __name__ == "__main__":
    main()
