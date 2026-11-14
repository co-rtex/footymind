"""Short-horizon defender movement forecasts against held-out tracking.

What this evaluates: how well each model forecasts where outfield defenders **actually went**
over the next 0.5 s and 1.0 s of real play, from a decision frame. It is *observational*.
It does not, and cannot, evaluate responses to hypothetical interventions: no ground truth
exists for those. A policy validated here is only shown to forecast real play better than
constant velocity.

Protocol (fixed before seeing results)
--------------------------------------
* Decision frames: start frames of attempted passes (``pass_dataset.label_events``, rules
  v2), deduplicated per frame.
* Targets: every outfield defender (the passer's opponents, goalkeeper excluded) whose
  position is present at ``t`` and at every frame up to ``t + 1.0 s`` in the same period,
  with a causal velocity at ``t``, no implausible-motion artefact in that window, and, on
  broadcast tracking, *detected* (not extrapolated) at ``t`` and at every target frame.
  Each exclusion reason is counted.
* Comparators: stationary; constant velocity (causal 0.2 s velocity); kinematically clipped
  constant velocity (the constant-velocity point projected onto the kinematic envelope,
  which equals constant velocity with the speed capped at the speed limit); the ``learned``
  ridge model (leave-one-match-out, ``RIDGE_ALPHA`` fixed a priori); the learned model
  clipped to the envelope (the variant the scenario engine serves).
* Metrics: ADE (mean error over every frame up to the horizon) and FDE (error at the
  horizon), in metres, in the attacking frame.
* Uncertainty: bootstrap over possessions (the clustering unit; frames and defenders within
  a possession are correlated), stratified by match, 1,000 resamples; paired differences.
* Envelope coverage: the share of real positions at the horizon inside the kinematic
  envelope (a check of the a priori constants; they are not tuned), plus a descriptive
  sensitivity over the reaction delay and the acceleration bound. In real play a defender
  is not reacting to a new stimulus at ``t``, so the reaction delay is expected to cost
  coverage here; for a response to an intervention it is the point of the delay.
* Pre-specified validation rule for the served ``learned`` policy: the 95% CI of
  FDE(learned, clipped) - FDE(constant velocity) lies below zero at both 0.5 s and 1.0 s.

Providers are never pooled. Run with ``python -m footymind.research.response_validation
<provider> [max_matches]``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import numpy.typing as npt

from footymind.counterfactual.kinematics import (
    DEFAULT_LIMITS,
    KinematicLimits,
    envelope_vertices,
    inside_polygon,
    prepare_velocity,
    project_onto_polygon,
)
from footymind.counterfactual.response_model import RIDGE_ALPHA, ResponseModel, features
from footymind.data.store import available_matches, git_revision, load_match
from footymind.paths import EXPERIMENTS_DIR
from footymind.research.pass_dataset import DatasetConfig, label_events
from footymind.spatial.state import causal_velocities, goalkeepers
from footymind.tracking.kinematics import implausible_motion_mask, velocity
from footymind.tracking.model import TeamSide

F = npt.NDArray[np.float64]

HORIZONS_S = (0.5, 1.0)
MAX_HORIZON_S = 1.0
N_BOOT = 1000
COMPARATORS = (
    "stationary",
    "const_velocity",
    "const_velocity_clipped",
    "learned",
    "learned_clipped",
)
ACCEL_SENSITIVITY = (2.5, 3.5, 5.0, 7.0)  # descriptive only; defaults are not tuned
REACTION_SENSITIVITY = (0.0, 0.25)
_CHUNK = 2000


def result_name(provider: str) -> str:
    return "response_validation_v1" + ("" if provider == "metrica" else f"_{provider}")


def horizon_frames(h: float, fps: float) -> int:
    return int(h * fps + 0.5)


def ball_velocities(match_id: str) -> F:
    """Causal (backward 0.2 s) ball velocity. Not speed-cleaned: passes exceed 12 m/s."""
    td = load_match(match_id).tracking
    return np.asarray(velocity(td.ball_xy, td.fps, "causal", 0.2, td.period), dtype=np.float64)


@dataclass
class Rows:
    match_id: list[str] = field(default_factory=list)
    group: list[str] = field(default_factory=list)
    frame: list[int] = field(default_factory=list)
    player_id: list[str] = field(default_factory=list)
    p: list[F] = field(default_factory=list)
    v: list[F] = field(default_factory=list)
    traj: list[F] = field(default_factory=list)
    xx: list[F] = field(default_factory=list)
    xy: list[F] = field(default_factory=list)


@dataclass
class Dataset:
    fps: float
    offsets_s: list[float]
    match_id: npt.NDArray[np.str_]
    group: npt.NDArray[np.str_]
    p: F  # (N, 2) attacking frame
    v: F  # (N, 2)
    traj: F  # (N, K, 2)
    xx: F
    xy: F
    excluded: dict[str, int]
    n_frames: int


def build_match(match_id: str, excluded: dict[str, int]) -> tuple[Rows, int, float]:
    b = load_match(match_id)
    td = b.tracking
    k_max = horizon_frames(MAX_HORIZON_S, td.fps)
    vel = causal_velocities(match_id).astype(np.float64)
    bvel = ball_velocities(match_id)
    gk = goalkeepers(match_id)
    art = implausible_motion_mask(td.xy, td.fps, period=td.period)
    passes = (
        label_events(b.events, DatasetConfig())
        .sort(["start_frame", "event_id"])
        .unique(subset=["start_frame"], keep="first", maintain_order=True)
    )
    rows = Rows()
    n_frames = 0

    def bump(k: str, n: int = 1) -> None:
        excluded[k] = excluded.get(k, 0) + n

    for r in passes.iter_rows(named=True):
        try:
            i = td.index_of_frame(int(r["start_frame"]))
        except KeyError:
            bump("frame_not_in_tracking")
            continue
        if i + k_max >= td.n_frames or td.period[i + k_max] != td.period[i]:
            bump("frames_horizon_crosses_period_end")
            continue
        ball = td.ball_xy[i].astype(np.float64)
        if np.isnan(ball).any():
            bump("frames_ball_missing")
            continue
        team: TeamSide = r["team"]
        other: TeamSide = "away" if team == "home" else "home"
        per = int(td.period[i])
        sign = td.attack_sign(team, per)
        defs = [j for j in td.team_indices(other) if j != gk[(per, other)]]
        atts = td.team_indices(team)
        n_frames += 1
        keep = []
        for j in defs:
            if np.isnan(td.xy[i, j]).any():
                continue  # not on the pitch: not a candidate target
            if np.isnan(vel[i, j]).any():
                bump("velocity_unavailable")
                continue
            if np.isnan(td.xy[i + 1 : i + k_max + 1, j]).any():
                bump("target_missing")
                continue
            if art[i : i + k_max + 1, j].any():
                bump("tracking_artefact")
                continue
            if td.detected is not None:
                if not td.detected[i, j]:
                    bump("input_extrapolated")
                    continue
                if not td.detected[i + 1 : i + k_max + 1, j].all():
                    bump("target_extrapolated")
                    continue
            keep.append(j)
        if not keep:
            continue
        idx = np.array(keep)
        p = td.xy[i, idx].astype(np.float64) * sign
        v = vel[i, idx] * sign
        xx, xy = features(
            p,
            v,
            ball * sign,
            bvel[i] * sign,
            td.xy[i, atts].astype(np.float64) * sign,
            td.pitch.half_length,
        )
        traj = np.transpose(td.xy[i + 1 : i + k_max + 1, idx].astype(np.float64), (1, 0, 2)) * sign
        pid = r["possession_id"] or f"none-{r['start_frame']}"
        for n, j in enumerate(keep):
            rows.match_id.append(match_id)
            rows.group.append(f"{match_id}|{pid}")
            rows.frame.append(int(r["start_frame"]))
            rows.player_id.append(td.players[j].player_id)
            rows.p.append(p[n])
            rows.v.append(v[n])
            rows.traj.append(traj[n])
            rows.xx.append(xx[n])
            rows.xy.append(xy[n])
    return rows, n_frames, td.fps


def build(match_ids: list[str]) -> Dataset:
    providers = {m.split("-", 1)[0] for m in match_ids}
    if len(providers) != 1:
        raise ValueError(f"refusing to pool providers {sorted(providers)}")
    excluded: dict[str, int] = {}
    parts = []
    fps_set = set()
    n_frames = 0
    for m in match_ids:
        rows, nf, fps = build_match(m, excluded)
        parts.append(rows)
        fps_set.add(fps)
        n_frames += nf
    if len(fps_set) != 1:
        raise ValueError(f"mixed frame rates {fps_set}")
    fps = fps_set.pop()
    k_max = horizon_frames(MAX_HORIZON_S, fps)

    def cat(attr: str) -> list[Any]:
        return [x for r in parts for x in getattr(r, attr)]

    return Dataset(
        fps=fps,
        offsets_s=[k / fps for k in range(1, k_max + 1)],
        match_id=np.array(cat("match_id")),
        group=np.array(cat("group")),
        p=np.array(cat("p")),
        v=np.array(cat("v")),
        traj=np.array(cat("traj")),
        xx=np.array(cat("xx")),
        xy=np.array(cat("xy")),
        excluded=excluded,
        n_frames=n_frames,
    )


def fit(
    ds: Dataset, mask: npt.NDArray[np.bool_] | None = None, meta: dict[str, Any] | None = None
) -> ResponseModel:
    m = np.ones(len(ds.p), dtype=bool) if mask is None else mask
    disp = ds.traj[m] - ds.p[m][:, None, :]
    return ResponseModel.fit(ds.xx[m], ds.xy[m], disp, ds.offsets_s, RIDGE_ALPHA, meta)


def _clip_all(p: F, v: F, pred: F, offsets: list[float], limits: KinematicLimits) -> F:
    """Project predictions ``(N, K, 2)`` onto each row's envelope at each offset."""
    out = np.empty_like(pred)
    vp, _, _ = prepare_velocity(v, limits)
    for a in range(0, len(p), _CHUNK):
        sl = slice(a, a + _CHUNK)
        verts = envelope_vertices(p[sl], vp[sl], offsets, limits)  # (K, n, H, 2)
        for k in range(len(offsets)):
            out[sl, k] = project_onto_polygon(verts[k], pred[sl, k])
    return out


def coverage(ds: Dataset, limits: KinematicLimits) -> dict[str, float]:
    vp, _, _ = prepare_velocity(ds.v, limits)
    res = {}
    for h in HORIZONS_S:
        k = horizon_frames(h, ds.fps)
        hits = []
        for a in range(0, len(ds.p), _CHUNK):
            sl = slice(a, a + _CHUNK)
            verts = envelope_vertices(ds.p[sl], vp[sl], [k / ds.fps], limits)[0]
            hits.append(inside_polygon(verts, ds.traj[sl, k - 1]))
        res[f"{h:.1f}s"] = float(np.concatenate(hits).mean())
    return res


def predictions(ds: Dataset, limits: KinematicLimits = DEFAULT_LIMITS) -> dict[str, F]:
    s = np.asarray(ds.offsets_s)[None, :, None]
    p = ds.p[:, None, :]
    vcap, _, _ = prepare_velocity(ds.v, limits)
    learned = np.full(ds.traj.shape, np.nan)
    for m in sorted(set(ds.match_id.tolist())):
        te = ds.match_id == m
        model = fit(ds, ~te)
        learned[te] = ds.p[te][:, None, :] + model.predict_all(ds.xx[te], ds.xy[te])
    return {
        "stationary": np.broadcast_to(p, ds.traj.shape).copy(),
        "const_velocity": p + ds.v[:, None, :] * s,
        "const_velocity_clipped": p + vcap[:, None, :] * s,
        "learned": learned,
        "learned_clipped": _clip_all(ds.p, ds.v, learned, ds.offsets_s, limits),
    }


def _bootstrap_weights(
    groups: npt.NDArray[np.int64], strata: list[npt.NDArray[np.int64]], n_boot: int, seed: int
) -> F:
    rng = np.random.default_rng(seed)
    n_groups = int(np.max(groups)) + 1
    w = np.zeros((n_boot, n_groups))
    for b in range(n_boot):
        for s in strata:
            w[b] += np.bincount(rng.choice(s, size=s.size), minlength=n_groups)
    return w


def evaluate(
    ds: Dataset, preds: dict[str, F], n_boot: int = N_BOOT, seed: int = 0
) -> dict[str, Any]:
    err = {c: np.linalg.norm(preds[c] - ds.traj, axis=-1) for c in COMPARATORS}  # (N, K)
    cols: dict[str, F] = {}
    for h in HORIZONS_S:
        k = horizon_frames(h, ds.fps)
        for c in COMPARATORS:
            cols[f"ade|{h}|{c}"] = err[c][:, :k].mean(axis=1)
            cols[f"fde|{h}|{c}"] = err[c][:, k - 1]
    names = list(cols)
    mat = np.column_stack([cols[n] for n in names])  # (N, M)
    uniq, ginv = np.unique(ds.group, return_inverse=True)
    sums = np.zeros((uniq.size, len(names)))
    np.add.at(sums, ginv, mat)
    counts = np.bincount(ginv, minlength=uniq.size).astype(float)
    g_match = np.array([u.split("|")[0] for u in uniq])
    strata = [np.flatnonzero(g_match == m) for m in np.unique(g_match)]
    w = _bootstrap_weights(np.arange(uniq.size), strata, n_boot, seed)
    boot = (w @ sums) / (w @ counts)[:, None]  # (B, M)
    point = mat.mean(axis=0)
    col = {n: i for i, n in enumerate(names)}

    def ci(a: F) -> list[float]:
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    out: dict[str, Any] = {}
    matches = sorted(set(ds.match_id.tolist()))
    for h in HORIZONS_S:
        k = horizon_frames(h, ds.fps)
        hk = f"{h:.1f}s"
        out[hk] = {"horizon_frames": k, "horizon_s_actual": k / ds.fps}
        for metric in ("ade", "fde"):
            block: dict[str, Any] = {}
            ref = col[f"{metric}|{h}|const_velocity"]
            for c in COMPARATORS:
                i = col[f"{metric}|{h}|{c}"]
                entry: dict[str, Any] = {"mean_m": float(point[i]), "ci95": ci(boot[:, i])}
                if c != "const_velocity":
                    d = boot[:, i] - boot[:, ref]
                    per_match = {
                        m: float(
                            mat[ds.match_id == m, i].mean() - mat[ds.match_id == m, ref].mean()
                        )
                        for m in matches
                    }
                    entry["minus_const_velocity"] = {
                        "delta_m": float(point[i] - point[ref]),
                        "ci95": ci(d),
                        "better_in_matches": int(sum(v < 0 for v in per_match.values())),
                        "n_matches": len(matches),
                        "per_match": per_match,
                    }
                block[c] = entry
            out[hk][metric] = block
    return out


def validated(summary: dict[str, Any]) -> bool:
    """Pre-specified rule: learned (clipped) beats constant velocity in FDE at every horizon."""
    return all(
        summary[f"{h:.1f}s"]["fde"]["learned_clipped"]["minus_const_velocity"]["ci95"][1] < 0
        for h in HORIZONS_S
    )


def main(
    provider: str = "metrica", max_matches: int | None = None, seed: int = 0
) -> dict[str, Any]:
    t0 = time.time()
    mids = [m for m in available_matches() if m.startswith(provider + "-")][:max_matches]
    ds = build(mids)
    preds = predictions(ds)
    summary = evaluate(ds, preds, seed=seed)
    cov: dict[str, Any] = {"default": coverage(ds, DEFAULT_LIMITS), "sensitivity": {}}
    for reaction in REACTION_SENSITIVITY:
        for accel in ACCEL_SENSITIVITY:
            lim = replace(DEFAULT_LIMITS, reaction_s=reaction, max_accel_mps2=accel)
            cov["sensitivity"][f"reaction={reaction},accel={accel}"] = coverage(ds, lim)
    speed = np.linalg.norm(ds.v, axis=1)
    name = result_name(provider)
    result = {
        "name": name,
        "provider": provider,
        "question": (
            "observational: forecast where outfield defenders actually went over the next "
            "0.5 s / 1.0 s from a pass-start frame; not a test of responses to hypothetical "
            "interventions (no ground truth exists)"
        ),
        "matches": mids,
        "code_revision": git_revision(),
        "runtime_s": None,
        "fps": ds.fps,
        "n_decision_frames": ds.n_frames,
        "n": len(ds.p),
        "n_possessions": int(np.unique(ds.group).size),
        "excluded_defender_rows": ds.excluded,
        "ridge_alpha": RIDGE_ALPHA,
        "kinematic_limits": {
            "reaction_s": DEFAULT_LIMITS.reaction_s,
            "max_accel_mps2": DEFAULT_LIMITS.max_accel_mps2,
            "max_speed_mps": DEFAULT_LIMITS.max_speed_mps,
        },
        "share_rows_speed_above_limit": float(np.mean(speed > DEFAULT_LIMITS.max_speed_mps)),
        "n_boot": N_BOOT,
        "bootstrap_unit": "possession (stratified by match)",
        "summary": summary,
        "envelope_coverage": cov,
        "validation_rule": (
            "learned policy validated iff the 95% CI of FDE(learned_clipped) - "
            "FDE(const_velocity) is below 0 at 0.5 s and at 1.0 s"
        ),
        "learned_validated": validated(summary),
    }
    result["runtime_s"] = round(time.time() - t0, 1)
    out = EXPERIMENTS_DIR / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    brief = {
        h: {m: {c: round(v["mean_m"], 3) for c, v in summary[h][m].items()} for m in ("ade", "fde")}
        for h in ("0.5s", "1.0s")
    }
    print(
        json.dumps(
            {
                "n": result["n"],
                "validated": result["learned_validated"],
                "summary": brief,
                "coverage": cov["default"],
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    import sys

    main(
        sys.argv[1] if len(sys.argv) > 1 else "metrica",
        max_matches=int(sys.argv[2]) if len(sys.argv) > 2 else None,
    )
