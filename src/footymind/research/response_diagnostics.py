"""Why is the learned response model inert? Controlled diagnostics and the innovation model.

``python -m footymind.research.response_diagnostics <provider> [max_matches]`` writes
``experiments/results/response_diagnostics_v1[_<provider>]/results.json``.

The ``learned`` model (``response_model.py``) forecasts real defender displacement better
than constant velocity (C40, C42), but a 5 m relocation of the carrier moves defenders by
centimetres (C44). Four questions, each answered on the ``response_validation`` rows
(pass-start frames, outfield defenders, detected-only on broadcast tracking, 1.0 s):

D1 *Conditioning.* Which of its inputs does an intervention change, and how much does the
   prediction change per metre of relocation?
D2 *Target.* How much of the displacement is own velocity (R^2 of constant velocity), and how
   much of the *innovation* (displacement beyond constant velocity) do its features explain?
D3 *Out of distribution.* Are the intervened feature vectors outside the training
   distribution (Mahalanobis distance against the observed rows' 99th percentile)? And how
   large is a 5 m relocation compared with real attacker innovations?
D4 *Innovation design.* Does a ridge model of defender innovation on attacker innovations
   (``counterfactual/response_innovation.py``) learn a non-trivial response?

Pre-registered rule for D4 (fixed before its first evaluation, session 4): leave-one-match-out,
ridge alpha = 1 a priori, every frame offset to 1.0 s. The innovation model is *supported* on a
provider iff at 0.5 s and at 1.0 s (i) the 95% CI of FDE(constant velocity + model) -
FDE(constant velocity) is below zero and (ii) the 95% CI of FDE(model) - FDE(placebo model)
is below zero, where the placebo model is refit after permuting attacker innovations among
the outfield attackers of each frame (team-wide movement and the ball kept, the
attacker-defender pairing broken). Possession-clustered bootstrap (1,000, stratified by
match), paired on the same rows. This tests a conditional association in real play, not a
causal response to interventions that never happened.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from footymind.counterfactual.response_innovation import (
    INNOVATION_FEATURES,
    InnovationResponseModel,
    innovation_features,
)
from footymind.counterfactual.response_model import (
    FEATURES_X,
    FEATURES_Y,
    RIDGE_ALPHA,
    ResponseModel,
    features,
    load_response_model,
)
from footymind.data.store import available_matches, git_revision, load_match
from footymind.paths import EXPERIMENTS_DIR
from footymind.research import response_validation as rv
from footymind.research.pass_dataset import DatasetConfig, label_events
from footymind.spatial.state import causal_velocities, goalkeepers

F = npt.NDArray[np.float64]

HORIZONS_S = (0.5, 1.0)
PROBE_M = 5.0
PROBE_DIRECTIONS = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
N_BOOT = 1000


def result_name(provider: str) -> str:
    return "response_diagnostics_v1" + ("" if provider == "metrica" else f"_{provider}")


@dataclass
class FrameTable:
    """Per decision frame, in the attacking frame: outfield attackers and the ball."""

    att_p: list[F]  # (m, 2) at t
    att_innov: list[F]  # (m, K, 2), NaN = unavailable
    ball_p: list[F]  # (2,)
    ball_v: list[F]  # (2,)
    ball_innov: list[F]  # (K, 2)
    carrier: list[int]  # index into att_p of the attacker nearest the ball (-1: none)
    excluded: dict[str, int]


def frame_table(ds: rv.Dataset) -> tuple[FrameTable, npt.NDArray[np.int64]]:
    """Attacker and ball innovations for every decision frame of ``ds``; row -> frame index."""
    k_n = len(ds.offsets_s)
    offs = np.asarray(ds.offsets_s)
    tab = FrameTable([], [], [], [], [], [], {})
    key_to_idx: dict[tuple[str, int], int] = {}
    for m in sorted(set(ds.match_id.tolist())):
        b = load_match(m)
        td = b.tracking
        vel = causal_velocities(m).astype(np.float64)
        bvel = rv.ball_velocities(m)
        gk = goalkeepers(m)
        passes = (
            label_events(b.events, DatasetConfig())
            .sort(["start_frame", "event_id"])
            .unique(subset=["start_frame"], keep="first", maintain_order=True)
        )
        team_of = dict(zip(passes["start_frame"].to_list(), passes["team"].to_list(), strict=True))
        for f in sorted(set(ds.frame[ds.match_id == m].tolist())):
            i = td.index_of_frame(int(f))
            team = team_of[int(f)]
            per = int(td.period[i])
            sign = td.attack_sign(team, per)
            att = [j for j in td.team_indices(team) if j != gk[(per, team)]]
            p = td.xy[i, att].astype(np.float64) * sign
            v = vel[i, att] * sign
            fut = np.transpose(td.xy[i + 1 : i + k_n + 1, att].astype(np.float64), (1, 0, 2)) * sign
            if td.detected is not None:
                ok = td.detected[i + 1 : i + k_n + 1, att].T & td.detected[i, att][:, None]
                fut[~ok] = np.nan
            innov = fut - p[:, None, :] - v[:, None, :] * offs[None, :, None]
            bp = td.ball_xy[i].astype(np.float64) * sign
            bv = bvel[i] * sign
            bfut = td.ball_xy[i + 1 : i + k_n + 1].astype(np.float64) * sign
            if td.ball_detected is not None:
                bok = td.ball_detected[i + 1 : i + k_n + 1] & td.ball_detected[i]
                bfut[~bok] = np.nan
            binnov = bfut - bp[None, :] - bv[None, :] * offs[:, None]
            d = np.linalg.norm(p - bp[None, :], axis=1)
            key_to_idx[(m, int(f))] = len(tab.att_p)
            tab.att_p.append(p)
            tab.att_innov.append(innov)
            tab.ball_p.append(bp)
            tab.ball_v.append(bv)
            tab.ball_innov.append(binnov)
            tab.carrier.append(int(np.nanargmin(d)) if np.isfinite(d).any() else -1)
    row_frame = np.array(
        [key_to_idx[(m, int(f))] for m, f in zip(ds.match_id, ds.frame, strict=True)]
    )
    return tab, row_frame


def build_features(
    ds: rv.Dataset, tab: FrameTable, row_frame: npt.NDArray[np.int64], placebo_seed: int | None
) -> F:
    """Innovation features ``(N, K, 5, 2)``; ``placebo_seed`` permutes attacker innovations."""
    rng = np.random.default_rng(placebo_seed) if placebo_seed is not None else None
    out = np.full((len(ds.p), len(ds.offsets_s), len(INNOVATION_FEATURES), 2), np.nan)
    for fi in range(len(tab.att_p)):
        rows = np.flatnonzero(row_frame == fi)
        if rows.size == 0:
            continue
        innov = tab.att_innov[fi]
        if rng is not None:
            innov = innov[rng.permutation(innov.shape[0])]
        out[rows] = innovation_features(ds.p[rows], tab.att_p[fi], innov, tab.ball_innov[fi])
    return out


def innovation_target(ds: rv.Dataset) -> F:
    offs = np.asarray(ds.offsets_s)[None, :, None]
    return np.asarray(ds.traj - ds.p[:, None, :] - ds.v[:, None, :] * offs)


def lomo_innovation(ds: rv.Dataset, feats: F, target: F) -> F:
    pred = np.full(target.shape, np.nan)
    for m in sorted(set(ds.match_id.tolist())):
        te = ds.match_id == m
        model = InnovationResponseModel.fit(feats[~te], target[~te], ds.offsets_s)
        pred[te] = model.predict_all(feats[te])
    return pred


def lomo_v1_on(ds: rv.Dataset, target: F) -> F:
    """The v1 feature set fit (LOMO) to an arbitrary target ``(N, K, 2)``."""
    pred = np.full(target.shape, np.nan)
    for m in sorted(set(ds.match_id.tolist())):
        te = ds.match_id == m
        model = ResponseModel.fit(ds.xx[~te], ds.xy[~te], target[~te], ds.offsets_s, RIDGE_ALPHA)
        pred[te] = model.predict_all(ds.xx[te], ds.xy[te])
    return pred


def r2(resid: F, centred_total: F) -> float:
    ok = np.isfinite(resid).all(axis=-1) & np.isfinite(centred_total).all(axis=-1)
    return float(1.0 - np.sum(resid[ok] ** 2) / np.sum(centred_total[ok] ** 2))


def _boot_means(ds: rv.Dataset, cols: dict[str, F], seed: int) -> dict[str, dict[str, Any]]:
    uniq, ginv = np.unique(ds.group, return_inverse=True)
    g_match = np.array([u.split("|")[0] for u in uniq])
    strata = [np.flatnonzero(g_match == m) for m in np.unique(g_match)]
    w = rv._bootstrap_weights(np.arange(uniq.size), strata, N_BOOT, seed)
    out = {}
    for name, x in cols.items():
        ok = np.isfinite(x)
        sums = np.bincount(ginv[ok], weights=x[ok], minlength=uniq.size)
        cnt = np.bincount(ginv[ok], minlength=uniq.size).astype(float)
        boot = (w @ sums) / np.maximum(w @ cnt, 1e-12)
        per_match = {
            m: float(x[ok & (ds.match_id == m)].mean()) for m in sorted(set(ds.match_id.tolist()))
        }
        out[name] = {
            "mean": float(x[ok].mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n": int(ok.sum()),
            "better_in_matches": int(sum(v < 0 for v in per_match.values())),
            "per_match": per_match,
        }
    return out


def probe_v1(
    ds: rv.Dataset, tab: FrameTable, row_frame: npt.NDArray[np.int64], provider: str
) -> dict[str, Any]:
    """D1 + D3 for the served v1 model: relocate the carrier (with the ball) by 5 m."""
    model = load_response_model(provider)
    if model is None:
        return {"available": False}
    k = model.offset_index(1.0)
    fx_obs, fy_obs = ds.xx, ds.xy
    zx = (np.where(np.isnan(fx_obs), model.x_medians, fx_obs) - model.x_mean) / model.x_scale
    zy = np.nan_to_num(fy_obs) / model.y_scale
    z_obs = np.column_stack([zx, zy])
    mu = z_obs.mean(axis=0)
    cov_inv = np.linalg.pinv(np.cov(z_obs, rowvar=False))

    def maha(z: F) -> F:
        d = z - mu
        return np.asarray(np.einsum("ni,ij,nj->n", d, cov_inv, d))

    d2_obs = maha(z_obs)
    p99 = float(np.percentile(d2_obs, 99))
    shifts_sd: dict[str, list[F]] = {
        n: [] for n in [*[f"{f}(x)" for f in FEATURES_X], *[f"{f}(y)" for f in FEATURES_Y]]
    }
    moves, max_moves, d2_hyp = [], [], []
    for fi in range(len(tab.att_p)):
        rows = np.flatnonzero(row_frame == fi)
        c = tab.carrier[fi]
        if rows.size == 0 or c < 0:
            continue
        m = ds.match_id[rows[0]]
        hl = load_match(m).tracking.pitch.half_length
        bv = tab.ball_v[fi]
        base = model.predict(
            *features(ds.p[rows], ds.v[rows], tab.ball_p[fi], bv, tab.att_p[fi], hl), k
        )
        for ux, uy in PROBE_DIRECTIONS:
            dlt = PROBE_M * np.array([ux, uy])
            att = tab.att_p[fi].copy()
            att[c] = att[c] + dlt
            xx, xy = features(ds.p[rows], ds.v[rows], tab.ball_p[fi] + dlt, bv, att, hl)
            pred = model.predict(xx, xy, k)
            mv = np.linalg.norm(pred - base, axis=1)
            moves.append(mv)
            max_moves.append(float(np.max(mv)))
            zxh = (np.where(np.isnan(xx), model.x_medians, xx) - model.x_mean) / model.x_scale
            zyh = np.nan_to_num(xy) / model.y_scale
            zh = np.column_stack([zxh, zyh])
            d2_hyp.append(maha(zh))
            for n, col in zip(shifts_sd, (zh - z_obs[rows]).T, strict=True):
                shifts_sd[n].append(np.abs(col))
    d2h = np.concatenate(d2_hyp)
    return {
        "available": True,
        "offset_s": model.offsets_s[k],
        "n_probes": len(max_moves),
        "mean_defender_move_m": float(np.concatenate(moves).mean()),
        "mean_largest_move_m": float(np.mean(max_moves)),
        "p95_largest_move_m": float(np.percentile(max_moves, 95)),
        "gain_m_per_m_largest": float(np.mean(max_moves) / PROBE_M),
        "mean_abs_feature_shift_sd": {
            n: float(np.concatenate(v).mean()) for n, v in shifts_sd.items()
        },
        "mahalanobis_sq_observed_p99": p99,
        "share_intervened_rows_beyond_observed_p99": float(np.mean(d2h > p99)),
        "share_observed_rows_beyond_observed_p99": float(np.mean(d2_obs > p99)),
        "coef_m_per_m_at_1s": {
            "x": dict(zip(FEATURES_X, (model.coef_x[k] / model.x_scale).tolist(), strict=True)),
            "y": dict(zip(FEATURES_Y, (model.coef_y[k] / model.y_scale).tolist(), strict=True)),
        },
    }


def probe_innovation(
    ds: rv.Dataset,
    tab: FrameTable,
    row_frame: npt.NDArray[np.int64],
    model: InnovationResponseModel,
) -> dict[str, Any]:
    """The innovation model's response to the same 5 m carrier relocation (unclipped)."""
    k = model.offset_index(1.0)
    max_moves, moves = [], []
    for fi in range(len(tab.att_p)):
        rows = np.flatnonzero(row_frame == fi)
        c = tab.carrier[fi]
        if rows.size == 0 or c < 0:
            continue
        m_att = tab.att_p[fi].shape[0]
        zero = np.zeros((m_att, 1, 2))
        base = model.predict(
            innovation_features(ds.p[rows], tab.att_p[fi], zero, np.zeros((1, 2)))[:, 0], k
        )
        for ux, uy in PROBE_DIRECTIONS:
            dlt = PROBE_M * np.array([ux, uy])
            iv = zero.copy()
            iv[c, 0] = dlt
            f = innovation_features(ds.p[rows], tab.att_p[fi], iv, dlt[None, :])[:, 0]
            mv = np.linalg.norm(model.predict(f, k) - base, axis=1)
            moves.append(mv)
            max_moves.append(float(np.max(mv)))
    return {
        "offset_s": model.offsets_s[k],
        "n_probes": len(max_moves),
        "mean_defender_move_m": float(np.concatenate(moves).mean()),
        "mean_largest_move_m": float(np.mean(max_moves)),
        "p95_largest_move_m": float(np.percentile(max_moves, 95)),
    }


def innovation_scale(ds: rv.Dataset, tab: FrameTable) -> dict[str, Any]:
    """D3 for the innovation design: how large are real attacker innovations?"""
    out = {}
    for h in HORIZONS_S:
        k = rv.horizon_frames(h, ds.fps) - 1
        a = np.concatenate([np.linalg.norm(iv[:, k], axis=-1) for iv in tab.att_innov])
        c = np.array(
            [
                np.linalg.norm(tab.att_innov[i][tab.carrier[i], k])
                for i in range(len(tab.att_p))
                if tab.carrier[i] >= 0
            ]
        )
        a, c = a[np.isfinite(a)], c[np.isfinite(c)]
        out[f"{h:.1f}s"] = {
            "attacker_innovation_m": {f"p{p}": float(np.percentile(a, p)) for p in (50, 90, 99)},
            "share_attacker_innovation_above_probe": float(np.mean(a > PROBE_M)),
            "carrier_innovation_m": {f"p{p}": float(np.percentile(c, p)) for p in (50, 90, 99)},
            "share_carrier_innovation_above_probe": float(np.mean(c > PROBE_M)),
        }
    return out


def main(
    provider: str = "metrica", max_matches: int | None = None, seed: int = 0
) -> dict[str, Any]:
    t0 = time.time()
    mids = [m for m in available_matches() if m.startswith(provider + "-")][:max_matches]
    ds = rv.build(mids)
    tab, row_frame = frame_table(ds)
    offs = np.asarray(ds.offsets_s)
    disp = ds.traj - ds.p[:, None, :]
    target = innovation_target(ds)
    feats = build_features(ds, tab, row_frame, None)
    feats_pl = build_features(ds, tab, row_frame, seed + 1)
    pred_in = lomo_innovation(ds, feats, target)
    pred_pl = lomo_innovation(ds, feats_pl, target)
    learned = ds.p[:, None, :] + lomo_v1_on(ds, disp)  # the LOMO v1 model, unclipped
    pred_v1_innov = lomo_v1_on(ds, target)
    # D2: variance shares at each horizon (rows with complete innovation features there)
    d2: dict[str, Any] = {}
    evals: dict[str, Any] = {}
    for h in HORIZONS_S:
        k = rv.horizon_frames(h, ds.fps) - 1
        ok = ~np.isnan(feats[:, k]).any(axis=(1, 2)) & ~np.isnan(feats_pl[:, k]).any(axis=(1, 2))
        dk = disp[ok, k]
        cen = dk - dk.mean(axis=0)
        tk = target[ok, k]
        d2[f"{h:.1f}s"] = {
            "n": int(ok.sum()),
            "r2_displacement_const_velocity": r2(tk, cen),
            "r2_displacement_learned_v1": r2(dk - (learned[ok, k] - ds.p[ok]), cen),
            "r2_innovation_v1_features": r2(tk - pred_v1_innov[ok, k], tk - tk.mean(axis=0)),
            "r2_innovation_innovation_model": r2(tk - pred_in[ok, k], tk - tk.mean(axis=0)),
            "r2_innovation_placebo_model": r2(tk - pred_pl[ok, k], tk - tk.mean(axis=0)),
            "rms_innovation_m": float(np.sqrt(np.mean(np.sum(tk**2, axis=1)))),
            "rms_velocity_part_m": float(
                np.sqrt(np.mean(np.sum((ds.v[ok] * offs[k]) ** 2, axis=1)))
            ),
        }
        e_cv = np.where(ok, np.linalg.norm(target[:, k], axis=1), np.nan)
        e_in = np.where(ok, np.linalg.norm(target[:, k] - pred_in[:, k], axis=1), np.nan)
        e_pl = np.where(ok, np.linalg.norm(target[:, k] - pred_pl[:, k], axis=1), np.nan)
        e_v1 = np.where(ok, np.linalg.norm(ds.traj[:, k] - learned[:, k], axis=1), np.nan)
        cols = {
            "fde_const_velocity": e_cv,
            "fde_innovation_model": e_in,
            "fde_placebo_model": e_pl,
            "fde_learned_v1": e_v1,
            "innovation_minus_cv": e_in - e_cv,
            "placebo_minus_cv": e_pl - e_cv,
            "innovation_minus_placebo": e_in - e_pl,
            "learned_v1_minus_cv": e_v1 - e_cv,
        }
        evals[f"{h:.1f}s"] = {"horizon_frames": k + 1} | _boot_means(ds, cols, seed)
    supported = all(
        evals[f"{h:.1f}s"]["innovation_minus_cv"]["ci95"][1] < 0
        and evals[f"{h:.1f}s"]["innovation_minus_placebo"]["ci95"][1] < 0
        for h in HORIZONS_S
    )
    full = InnovationResponseModel.fit(feats, target, ds.offsets_s)
    k1 = full.offset_index(1.0)
    k05 = full.offset_index(0.5)
    gains = full.gains_m_per_m()
    coefs = {
        f"{full.offsets_s[kk]:.2f}s": {
            ax: dict(zip(INNOVATION_FEATURES, gains[ax][kk].tolist(), strict=True))
            for ax in ("x", "y")
        }
        for kk in (k05, k1)
    }
    full_pl = InnovationResponseModel.fit(feats_pl, target, ds.offsets_s)
    gpl = full_pl.gains_m_per_m()
    coefs_pl = {
        ax: dict(zip(INNOVATION_FEATURES, gpl[ax][k1].tolist(), strict=True)) for ax in ("x", "y")
    }
    v1_on_innov = ResponseModel.fit(ds.xx, ds.xy, target, ds.offsets_s, RIDGE_ALPHA)
    kv = v1_on_innov.offset_index(1.0)
    name = result_name(provider)
    result: dict[str, Any] = {
        "name": name,
        "provider": provider,
        "question": "why does the learned response model barely respond to interventions, and "
        "does an innovation model of defender movement on attacker movement respond?",
        "matches": mids,
        "code_revision": git_revision(),
        "runtime_s": None,
        "fps": ds.fps,
        "n": len(ds.p),
        "n_decision_frames": len(tab.att_p),
        "n_possessions": int(np.unique(ds.group).size),
        "excluded_defender_rows": ds.excluded,
        "ridge_alpha": RIDGE_ALPHA,
        "n_boot": N_BOOT,
        "d1_d3_probe_learned_v1": probe_v1(ds, tab, row_frame, provider),
        "d2_variance": d2,
        "d2_v1_features_on_innovation_coef_m_per_m_at_1s": {
            "x": dict(
                zip(
                    FEATURES_X, (v1_on_innov.coef_x[kv] / v1_on_innov.x_scale).tolist(), strict=True
                )
            ),
            "y": dict(
                zip(
                    FEATURES_Y, (v1_on_innov.coef_y[kv] / v1_on_innov.y_scale).tolist(), strict=True
                )
            ),
        },
        "d3_innovation_scale": innovation_scale(ds, tab),
        "d4_innovation_model": {
            "features": INNOVATION_FEATURES,
            "evaluation": evals,
            "rule": "supported iff, at 0.5 s and 1.0 s, the 95% CIs of FDE(cv + model) - FDE(cv) "
            "and of FDE(model) - FDE(placebo model) are below 0 (LOMO, paired rows)",
            "supported": supported,
            "gains_m_per_m_all_data": coefs,
            "placebo_gains_m_per_m_all_data_1s": coefs_pl,
            "probe_5m_carrier_unclipped": probe_innovation(ds, tab, row_frame, full),
        },
    }
    result["runtime_s"] = round(time.time() - t0, 1)
    out = EXPERIMENTS_DIR / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("n", "d2_variance")}, indent=1))
    print(
        json.dumps(
            {
                h: {c: (v["mean"], v["ci95"]) for c, v in e.items() if isinstance(v, dict)}
                for h, e in evals.items()
            },
            indent=1,
        )
    )
    print("supported", supported)
    return result


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "metrica",
        max_matches=int(sys.argv[2]) if len(sys.argv) > 2 else None,
    )
