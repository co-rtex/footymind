"""Off-ball opportunity under reactive defenders: pre-registered study (session 4).

``python -m footymind.research.offball_study <provider> [n_per_match]`` writes
``experiments/results/offball_opportunity_v1[_<provider>]/results.json``. Providers are never
pooled. Definitions are in ``ghost/offball_opportunity.py``; the note with the results is
``docs/research/offball_opportunity.md``.

Pre-registration (written and committed before the study was run on full data; only a
2-match pilot was run to check runtime and code paths, and its numbers are not used)
---------------------------------------------------------------------------------------
*Question.* For an attacker's observed off-ball movement over the last 1.0 s before a pass
decision, read at the 0.2 s guard frame (ADR-0007), how much does the team's modelled passing
opportunity change once plausible defender responses to that movement are accounted for, and
how much does the answer depend on the response assumption?

*Samples.* Attempted passes (rules v2), ``n_per_match`` per match (seed 0): 150 on Metrica
(3 matches) and 100 on SkillCorner (20 matches; fixed for runtime, about 1 s per frame).
The analysed frame ``t`` is 0.2 s before the event start; the earlier frame is ``t - 1.0 s``
in the same period. The attacking team must match the event's team, with a carrier and an
observed ball at ``t``. Attackers: every off-ball outfield attacker on the pitch with an
earlier position. On SkillCorner the primary set keeps attackers *detected* at ``t`` and
``t - 1 s`` (an extrapolated "run" is not an observed run); all attackers are a sensitivity
set. The analysis
set is the complete cases (every assumption and the placebo defined for the primary measure);
exclusions are counted.

*Measures.* ``forward`` (primary; best progressive option), ``sum`` and ``own``.
*Assumptions.* ``static``, ``constant_velocity``, ``marker_follows``, ``lane_cover`` (policy
response to the attacker's innovation, primary); ``full`` attribution is a sensitivity.

*Hypotheses and decision rules* (95% possession-clustered bootstrap CIs, 1,000 resamples;
SkillCorner also reports a two-stage match-level CI):

* **H-R1 (response adjustment).** Under ``marker_follows`` and ``lane_cover`` the mean change
  differs from ``static``: CI of mean(change_k - change_static) excludes 0. Expected sign:
  positive. Removing a response puts the responding defender back near the attacker's earlier
  position or lane, so the run is credited with the space it dragged defenders out of.
* **H-R2 (assumption dependence).** The answer depends materially on the assumption if the
  mean between-assumption range (max - min over the four assumptions, per attacker) is at least
  25% of the mean |static change| (lower CI bound >= 0.25; the same bar as H-G2).
* **H-R3 (null).** A rotation placebo keeps each attacker's displacement size (and innovation
  size) and randomises its direction. Real runs are directed toward opportunity if
  mean(change_real - change_placebo) > 0 (CI) under each assumption.
* **Ranking agreement (descriptive).** Within frames with >= 3 analysed attackers: mean
  Spearman correlation between ``static`` and each other assumption, and the share of frames
  with the same top-ranked attacker. The ranking is called assumption-robust if every top-1
  agreement is >= 0.8.
* **O1 (outcome: receiver).** On completed passes, rank the analysed attackers by the ``own``
  change; top-1 accuracy for the actual receiver (the receiver id is a label only, never an
  input). Response-adjusted is better than static iff the CI of the accuracy difference > 0.
  Expected: not better (run value was barely above chance, docs/RESEARCH.md).
* **O2 (outcome: possession progress).** Progress = the largest forward ball movement
  (attacking frame, metres) from ``t`` over the next 10 s while the possession lasts, on
  observed (detected) ball frames only (NaN if none). Statistic: Spearman correlation between
  the frame's summed ``forward`` change over its analysed attackers and progress. Response-
  adjusted is better than static iff the CI of the correlation difference > 0.
* *Descriptive:* per-assumption distributions (mean, median, p10, p90, share > 0, mean |x|),
  the ``full`` attribution next to ``innovation``, a runners subset (displacement >= 3 m).

Interpretation limits: model-based, mixed states never happened, response rules are
assumptions with a necessary observational check only (C37). Not causal, not player value.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from footymind.counterfactual.response_innovation import load_innovation_model
from footymind.data.store import available_matches, git_revision, load_match
from footymind.ghost.offball_opportunity import (
    ASSUMPTIONS,
    CONTROL_ASSUMPTIONS,
    MEASURES,
    POLICY_ASSUMPTIONS,
    eligible_attackers,
    offball_opportunity,
    rotate_history,
)
from footymind.paths import EXPERIMENTS_DIR
from footymind.research.pass_dataset import DatasetConfig, _prev_event_features, label_events
from footymind.spatial.state import causal_velocities, frame_state

HORIZON_S = 1.0
GUARD_S = 0.2
PROGRESS_WINDOW_S = 10.0
RUNNER_MIN_M = 3.0
SPREAD_BAR = 0.25
TOP1_ROBUST = 0.8


def _progress(b: Any, i: int, team: str, possession_id: str | None) -> float:
    """Largest forward ball movement over the next window while the possession lasts."""
    td = b.tracking
    if possession_id is None:
        return np.nan
    pos = b.possessions.filter(pl.col("possession_id") == possession_id)
    if pos.height == 0:
        return np.nan
    end_frame = int(pos["end_frame"][0])
    sign = td.attack_sign(team, int(td.period[i]))
    hi = min(i + round(PROGRESS_WINDOW_S * td.fps), td.n_frames - 1)
    js = np.arange(i + 1, hi + 1)
    js = js[(td.frame[js] <= end_frame) & (td.period[js] == td.period[i])]
    if js.size == 0 or np.isnan(td.ball_xy[i]).any():
        return np.nan
    bx = td.ball_xy[js, 0].astype(np.float64)
    ok = ~np.isnan(bx)
    if td.ball_detected is not None:
        ok &= td.ball_detected[js]
        if not bool(td.ball_detected[i]):
            return np.nan
    if not ok.any():
        return np.nan
    return float(np.max((bx[ok] - float(td.ball_xy[i, 0])) * sign))


def _rows(
    provider: str, n_per_match: int, seed: int, matches: list[str] | None = None
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {"passes_sampled": 0, "frames_used": 0, "skipped_history": 0, "skipped_state": 0}
    rng = np.random.default_rng(seed)
    innovation_model = load_innovation_model(provider)
    if innovation_model is None:
        raise ValueError(f"no innovation response model for {provider} (positive control)")
    mids = matches or [m for m in available_matches() if m.startswith(provider + "-")]
    for mid in mids:
        if not mid.startswith(provider + "-"):
            raise ValueError(f"{mid} is not a {provider} match (providers are never pooled)")
        b = load_match(mid)
        td = b.tracking
        vel = causal_velocities(mid)
        k = round(HORIZON_S * td.fps)
        g = round(GUARD_S * td.fps)
        ev = label_events(_prev_event_features(b.events), DatasetConfig())
        ev = ev.sample(min(n_per_match, ev.height), seed=seed)
        for r in ev.iter_rows(named=True):
            counts["passes_sampled"] += 1
            i = td.index_of_frame(int(r["start_frame"])) - g
            if i - k < 0 or td.period[i - k] != td.period[i]:
                counts["skipped_history"] += 1
                continue
            st = frame_state(mid, int(td.frame[i]))
            if st.attacking_team != r["team"] or st.carrier is None or np.isnan(st.ball).any():
                counts["skipped_state"] += 1
                continue
            exy = td.xy[i - k].astype(np.float64)
            evel = vel[i - k].astype(np.float64)
            att = eligible_attackers(st, exy)
            if not att:
                counts["skipped_state"] += 1
                continue
            counts["frames_used"] += 1
            real = offball_opportunity(st, exy, evel, HORIZON_S, "innovation", td.pitch, att)
            full = offball_opportunity(
                st, exy, evel, HORIZON_S, "full", td.pitch, att, POLICY_ASSUMPTIONS
            )
            ctrl = offball_opportunity(
                st,
                exy,
                evel,
                HORIZON_S,
                "innovation",
                td.pitch,
                att,
                CONTROL_ASSUMPTIONS,
                innovation_model=innovation_model,
            )
            pxy, pvel = rotate_history(st, exy, evel, att, rng)
            plc = offball_opportunity(st, pxy, pvel, HORIZON_S, "innovation", td.pitch, att)
            progress = _progress(b, i, r["team"], r["possession_id"])
            det_now = td.detected[i] if td.detected is not None else None
            det_then = td.detected[i - k] if td.detected is not None else None
            for n, a in enumerate(att):
                ra, fa, pa = real.attackers[n], full.attackers[n], plc.attackers[n]
                row: dict[str, Any] = {
                    "match_id": mid,
                    "possession_id": r["possession_id"] or "none",
                    "frame": int(td.frame[i]),
                    "event_id": int(r["event_id"]),
                    "success": int(r["success"]),
                    "player_id": ra.player_id,
                    "is_receiver": int(r["success"] == 1 and r["to_player_id"] == ra.player_id),
                    "detected_both": bool(det_now[a] and det_then[a])
                    if det_now is not None and det_then is not None
                    else True,
                    "displacement_m": ra.displacement_m,
                    "innovation_m": ra.innovation_m,
                    "progress_m": progress,
                    "v_forward_obs": real.factual["observed"]["forward"],
                }
                for kk in ASSUMPTIONS:
                    for m in MEASURES:
                        row[f"{kk}__{m}"] = ra.change[kk][m]
                        row[f"placebo_{kk}__{m}"] = pa.change[kk][m]
                ca = ctrl.attackers[n]
                for kk in CONTROL_ASSUMPTIONS:
                    row[f"{kk}__moved"] = ca.defenders_moved[kk]
                    for m in MEASURES:
                        row[f"control_{kk}__{m}"] = ca.change[kk][m]
                for kk in POLICY_ASSUMPTIONS:
                    row[f"{kk}__moved"] = ra.defenders_moved[kk]
                    for m in MEASURES:
                        row[f"full_{kk}__{m}"] = fa.change[kk][m]
                rows.append(row)
    return rows, counts


# ---------------------------------------------------------------------------------------
# Bootstrap over (match, possession) clusters, by cluster sums (fast for mean-type stats)


class Clusters:
    def __init__(self, df: pl.DataFrame) -> None:
        keys = (df["match_id"] + "|" + df["possession_id"]).to_numpy()
        uniq, self.inv = np.unique(keys, return_inverse=True)
        self.g = int(uniq.size)
        cm = np.array([k.split("|", 1)[0] for k in uniq])
        self.matches = sorted(set(cm))
        self.by_match = [np.flatnonzero(cm == m) for m in self.matches]

    def draws(self, n_boot: int, seed: int, two_stage: bool) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n_boot):
            if two_stage:
                ms = rng.integers(0, len(self.matches), len(self.matches))
                out.append(
                    np.concatenate(
                        [
                            self.by_match[m][
                                rng.integers(0, self.by_match[m].size, self.by_match[m].size)
                            ]
                            for m in ms
                        ]
                    )
                )
            else:
                out.append(rng.integers(0, self.g, self.g))
        return out


def _sum_stat(
    df: pl.DataFrame,
    cols: list[np.ndarray],
    stat: Callable[[np.ndarray, float], float],
    cl: Clusters,
    draws: list[np.ndarray],
) -> dict[str, Any]:
    vals = np.column_stack(cols)
    s = np.zeros((cl.g, vals.shape[1]))
    np.add.at(s, cl.inv, vals)
    c = np.bincount(cl.inv, minlength=cl.g).astype(float)
    point = stat(s.sum(axis=0), float(c.sum()))
    boots = [stat(s[d].sum(axis=0), float(c[d].sum())) for d in draws]
    return {
        "value": point,
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
    }


def _mean(df: pl.DataFrame, x: np.ndarray, cl: Clusters, draws: list[np.ndarray]) -> dict[str, Any]:
    return _sum_stat(df, [x], lambda t, n: float(t[0] / n), cl, draws)


def _ratio(
    df: pl.DataFrame, a: np.ndarray, b: np.ndarray, cl: Clusters, draws: list[np.ndarray]
) -> dict[str, Any]:
    return _sum_stat(df, [a, b], lambda t, n: float(t[0] / max(t[1], 1e-12)), cl, draws)


def _distribution(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
        "share_positive": float((x > 1e-9).mean()),
        "share_negative": float((x < -1e-9).mean()),
        "mean_abs": float(np.abs(x).mean()),
    }


def _col(df: pl.DataFrame, name: str) -> np.ndarray:
    return df[name].to_numpy().astype(np.float64)


def analyse(df: pl.DataFrame, n_boot: int, seed: int, two_stage: bool) -> dict[str, Any]:
    """Every pre-registered statistic on one analysis set."""
    cl = Clusters(df)
    draws = cl.draws(n_boot, seed, two_stage)
    out: dict[str, Any] = {
        "n_attacker_rows": df.height,
        "n_frames": int(df.select("match_id", "frame").n_unique()),
        "n_clusters": cl.g,
        "n_matches": len(cl.matches),
    }
    for m in MEASURES:
        res: dict[str, Any] = {}
        st = _col(df, f"static__{m}")
        for k in ASSUMPTIONS:
            x = _col(df, f"{k}__{m}")
            res[k] = {"distribution": _distribution(x), "mean": _mean(df, x, cl, draws)}
            if k != "static":
                res[k]["minus_static"] = _mean(df, x - st, cl, draws)
            res[k]["real_minus_placebo"] = _mean(df, x - _col(df, f"placebo_{k}__{m}"), cl, draws)
            if k != "static":
                adj_real = x - st
                adj_plc = _col(df, f"placebo_{k}__{m}") - _col(df, f"placebo_static__{m}")
                res[k]["adjustment_real_minus_placebo"] = _mean(df, adj_real - adj_plc, cl, draws)
        for k in POLICY_ASSUMPTIONS:
            fx = _col(df, f"full_{k}__{m}")
            ok = np.isfinite(fx)
            sub = df.filter(pl.Series(ok))
            if sub.height > 30:
                c2 = Clusters(sub)
                d2 = c2.draws(n_boot, seed, two_stage)
                res[f"full_{k}"] = {
                    "n": int(sub.height),
                    "mean": _mean(sub, fx[ok], c2, d2),
                    "minus_static": _mean(sub, fx[ok] - st[ok], c2, d2),
                    "minus_innovation": _mean(sub, fx[ok] - _col(sub, f"{k}__{m}"), c2, d2),
                }
        for k in CONTROL_ASSUMPTIONS:  # positive control (post-review), not pre-registered
            cx = _col(df, f"control_{k}__{m}")
            ok = np.isfinite(cx)
            sub = df.filter(pl.Series(ok))
            if sub.height > 30:
                c3 = Clusters(sub)
                d3 = c3.draws(n_boot, seed, two_stage)
                res[f"control_{k}"] = {
                    "n": int(sub.height),
                    "mean": _mean(sub, cx[ok], c3, d3),
                    "minus_static": _mean(sub, cx[ok] - st[ok], c3, d3),
                    "mean_defenders_moved": float(_col(sub, f"{k}__moved").mean()),
                }
        allv = np.column_stack([_col(df, f"{k}__{m}") for k in ASSUMPTIONS])
        rng_all = allv.max(axis=1) - allv.min(axis=1)
        pol = np.column_stack([_col(df, f"{k}__{m}") for k in ("static", *POLICY_ASSUMPTIONS)])
        rng_pol = pol.max(axis=1) - pol.min(axis=1)
        res["spread"] = {
            "mean_range_all_four": float(rng_all.mean()),
            "range_over_mean_abs_static": _ratio(df, rng_all, np.abs(st), cl, draws),
            "range_static_and_policies_over_mean_abs_static": _ratio(
                df, rng_pol, np.abs(st), cl, draws
            ),
            "sign_disagreement_share": float(
                ((allv > 1e-9).any(axis=1) & (allv < -1e-9).any(axis=1)).mean()
            ),
        }
        out[m] = res
    p = out["forward"]["spread"]["range_over_mean_abs_static"]
    out["hypotheses"] = {
        "H-R1": {
            k: bool(
                out["forward"][k]["minus_static"]["ci95"][0] > 0
                or out["forward"][k]["minus_static"]["ci95"][1] < 0
            )
            for k in POLICY_ASSUMPTIONS
        },
        "H-R1_direction_positive": {
            k: bool(out["forward"][k]["minus_static"]["ci95"][0] > 0) for k in POLICY_ASSUMPTIONS
        },
        "H-R2": bool(p["ci95"][0] >= SPREAD_BAR),
        "H-R3": {
            k: bool(out["forward"][k]["real_minus_placebo"]["ci95"][0] > 0) for k in ASSUMPTIONS
        },
    }
    out["ranking"] = _ranking(df, n_boot, seed, two_stage)
    out["outcomes"] = _outcomes(df, n_boot, seed, two_stage)
    return out


def _frames(df: pl.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    keys = (df["match_id"] + "|" + df["frame"].cast(pl.Utf8)).to_numpy()
    uniq, inv = np.unique(keys, return_inverse=True)
    mids, poss = df["match_id"].to_numpy(), df["possession_id"].to_numpy()
    out = []
    for g in range(uniq.size):
        idx = np.flatnonzero(inv == g)
        out.append((str(mids[idx[0]]), str(poss[idx[0]]), idx))
    return out


def _frame_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None)


def _ranking(df: pl.DataFrame, n_boot: int, seed: int, two_stage: bool) -> dict[str, Any]:
    fr = [f for f in _frames(df) if f[2].size >= 3]
    out: dict[str, Any] = {"n_frames_ge3_attackers": len(fr)}
    for m in ("forward", "own"):
        st = _col(df, f"static__{m}")
        res: dict[str, Any] = {}
        for k in ASSUMPTIONS[1:]:
            x = _col(df, f"{k}__{m}")
            rows = []
            for mid, pid, idx in fr:
                a, b = st[idx], x[idx]
                rho = (
                    float(spearmanr(a, b).statistic)
                    if np.ptp(a) > 1e-12 and np.ptp(b) > 1e-12
                    else np.nan
                )
                rows.append(
                    {
                        "match_id": mid,
                        "possession_id": pid,
                        "rho": rho,
                        "top1_same": float(int(np.argmax(a)) == int(np.argmax(b))),
                    }
                )
            f = _frame_df(rows)
            fr_ok = f.filter(pl.col("rho").is_not_nan())
            cl, cl2 = Clusters(f), Clusters(fr_ok)
            res[k] = {
                "mean_spearman": _mean(
                    fr_ok, _col(fr_ok, "rho"), cl2, cl2.draws(n_boot, seed, two_stage)
                ),
                "n_frames_defined_spearman": fr_ok.height,
                "top1_agreement": _mean(
                    f, _col(f, "top1_same"), cl, cl.draws(n_boot, seed, two_stage)
                ),
            }
        res["assumption_robust"] = bool(
            all(res[k]["top1_agreement"]["value"] >= TOP1_ROBUST for k in ASSUMPTIONS[1:])
        )
        out[m] = res
    return out


def _outcomes(df: pl.DataFrame, n_boot: int, seed: int, two_stage: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # O1: receiver identification on completed passes whose receiver is an analysed attacker
    comp = df.filter(pl.col("success") == 1)
    fr = [f for f in _frames(comp) if comp["is_receiver"].to_numpy()[f[2]].sum() == 1]
    rec = comp["is_receiver"].to_numpy()
    rows = []
    for mid, pid, idx in fr:
        row: dict[str, Any] = {"match_id": mid, "possession_id": pid, "chance": 1.0 / idx.size}
        for k in ASSUMPTIONS:
            v = _col(comp, f"{k}__own")[idx]
            row[k] = float(rec[idx][int(np.argmax(v))] == 1)
        rows.append(row)
    o1: dict[str, Any] = {"n_frames": len(rows)}
    if len(rows) > 30:
        f = _frame_df(rows)
        cl = Clusters(f)
        d = cl.draws(n_boot, seed, two_stage)
        o1["chance"] = float(f["chance"].mean())  # type: ignore[arg-type]
        for k in ASSUMPTIONS:
            o1[k] = _mean(f, _col(f, k), cl, d)
            if k != "static":
                o1[f"{k}_minus_static"] = _mean(f, _col(f, k) - _col(f, "static"), cl, d)
        o1["response_adjusted_better"] = {
            k: bool(o1[f"{k}_minus_static"]["ci95"][0] > 0) for k in POLICY_ASSUMPTIONS
        }
    out["O1_receiver_top1"] = o1
    # O2: frame-level summed forward change vs possession progress (Spearman)
    frames = _frames(df)
    prog = _col(df, "progress_m")
    rows = []
    for mid, pid, idx in frames:
        if not np.isfinite(prog[idx[0]]):
            continue
        row = {"match_id": mid, "possession_id": pid, "progress": float(prog[idx[0]])}
        for k in ASSUMPTIONS:
            row[k] = float(_col(df, f"{k}__forward")[idx].sum())
        rows.append(row)
    o2: dict[str, Any] = {"n_frames": len(rows)}
    if len(rows) > 30:
        f = _frame_df(rows)
        cl = Clusters(f)
        y = _col(f, "progress")
        xs = {k: _col(f, k) for k in ASSUMPTIONS}
        members = [np.flatnonzero(cl.inv == g) for g in range(cl.g)]

        def rho(x: np.ndarray, idx: np.ndarray) -> float:
            return float(spearmanr(x[idx], y[idx]).statistic)

        boots: dict[str, list[float]] = {k: [] for k in ASSUMPTIONS}
        diffs: dict[str, list[float]] = {k: [] for k in (*POLICY_ASSUMPTIONS, "constant_velocity")}
        for dr in cl.draws(n_boot, seed, two_stage):
            idx = np.concatenate([members[g] for g in dr])
            r = {k: rho(xs[k], idx) for k in ASSUMPTIONS}
            for k in ASSUMPTIONS:
                boots[k].append(r[k])
            for k in diffs:
                diffs[k].append(r[k] - r["static"])
        allidx = np.arange(f.height)
        for k in ASSUMPTIONS:
            o2[k] = {
                "spearman": rho(xs[k], allidx),
                "ci95": [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))],
            }
        for k, v in diffs.items():
            o2[f"{k}_minus_static"] = {
                "value": rho(xs[k], allidx) - rho(xs["static"], allidx),
                "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            }
        o2["response_adjusted_better"] = {
            k: bool(o2[f"{k}_minus_static"]["ci95"][0] > 0) for k in POLICY_ASSUMPTIONS
        }
    out["O2_progress_spearman"] = o2
    return out


def complete_cases(df: pl.DataFrame) -> pl.DataFrame:
    cols = [f"{k}__forward" for k in ASSUMPTIONS] + [f"placebo_{k}__forward" for k in ASSUMPTIONS]
    cols += [f"{k}__{m}" for k in ASSUMPTIONS for m in ("sum", "own")]
    return df.filter(pl.all_horizontal([pl.col(c).is_finite() for c in cols]))


def main(
    provider: str = "metrica",
    n_per_match: int = 150,
    seed: int = 0,
    n_boot: int = 1000,
    matches: list[str] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    rows, counts = _rows(provider, n_per_match, seed, matches)
    df = pl.DataFrame(rows, infer_schema_length=None)
    cc = complete_cases(df)
    primary = cc.filter(pl.col("detected_both")) if provider != "metrica" else cc
    res: dict[str, Any] = {
        "name": "offball_opportunity_v1" + ("" if provider == "metrica" else f"_{provider}"),
        "provider": provider,
        "code_revision": git_revision(),
        "horizon_s": HORIZON_S,
        "guard_s": GUARD_S,
        "n_per_match": n_per_match,
        "seed": seed,
        "n_boot": n_boot,
        "measure_primary": "forward (best progressive option, > 5 m ahead of the ball)",
        "attribution_primary": "innovation",
        "counts": {
            **counts,
            "attacker_rows": df.height,
            "complete_case_rows": cc.height,
            "excluded_incomplete": df.height - cc.height,
            "primary_rows": primary.height,
        },
        "primary_set": "complete cases"
        + ("" if provider == "metrica" else ", attacker detected at t and t - 1 s"),
        "primary": analyse(primary, n_boot, seed, two_stage=False),
    }
    runners = primary.filter(pl.col("displacement_m") >= RUNNER_MIN_M)
    if runners.height > 100:
        res["runners_subset"] = {
            "rule": f"displacement >= {RUNNER_MIN_M} m in {HORIZON_S} s",
            **analyse(runners, n_boot, seed, two_stage=False),
        }
    if provider != "metrica":
        res["primary_match_level"] = analyse(primary, n_boot, seed, two_stage=True)
        res["all_attackers_sensitivity"] = analyse(cc, n_boot, seed, two_stage=False)
    res["runtime_s"] = round(time.time() - t0, 1)
    if write:
        out = EXPERIMENTS_DIR / "results" / res["name"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


N_PER_MATCH = {"metrica": 150, "skillcorner": 100}

if __name__ == "__main__":
    prov = sys.argv[1] if len(sys.argv) > 1 else "metrica"
    r = main(prov, int(sys.argv[2]) if len(sys.argv) > 2 else N_PER_MATCH.get(prov, 100))
    print(json.dumps({"counts": r["counts"], "hypotheses": r["primary"]["hypotheses"]}, indent=2))
