"""Stability and redundancy study for Defensive Shadow.

Run with ``python -m footymind.research.shadow_stability``.

Questions:
1. **Noise robustness.** Does the per-frame ranking of defenders survive realistic
   position noise? Positions are perturbed with isotropic Gaussian noise (σ = 0.1 m,
   roughly optical-tracking precision, and σ = 0.3 m, a pessimistic value), 3 draws each.
2. **Ring-radius sensitivity.** Are results stable for ring_fraction 0.3 / 0.5 / 0.7?
3. **Redundancy.** Is the metric just a re-expression of a simple quantity? It is compared
   with the summed heuristic interception probability of the defender over all current
   pass lanes ("lane presence").

Metric: Spearman rank correlation across the outfield defenders of a frame, averaged over
frames. Frames: a fixed random sample of pass starts from each match (seeded).
"""

from __future__ import annotations

import json
import time

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from footymind.data.store import git_revision, load_match
from footymind.ghost.defensive_shadow import shadow_for_state
from footymind.paths import EXPERIMENTS_DIR
from footymind.spatial.lanes import lane
from footymind.spatial.state import FrameState, frame_state


def lane_presence(st: FrameState) -> dict[str, float]:
    assert st.attacking_team is not None
    att = st.team_mask(st.attacking_team) & st.on_pitch()
    dfn = (~st.team_mask(st.attacking_team)) & st.on_pitch()
    didx = np.flatnonzero(dfn)
    tot = np.zeros(didx.size)
    for r in np.flatnonzero(att):
        if r == st.carrier:
            continue
        tot += lane(st.ball, st.xy[r], st.xy[didx], st.vel[didx]).p_intercept_by_defender
    return {st.player_ids[d]: float(v) for d, v in zip(didx, tot, strict=True)}


def _sum(st: FrameState, **kw: float) -> dict[str, float]:
    return {d.player_id: d.shadow_sum for d in shadow_for_state(st, **kw).defenders}  # type: ignore[arg-type]


def _rho(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) & set(b))
    if len(keys) < 4:
        return float("nan")
    r = spearmanr([a[k] for k in keys], [b[k] for k in keys]).statistic
    return float(r)


def main(n_per_match: int = 15, seed: int = 0) -> None:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    rows = []
    for mid in ("metrica-sample-1", "metrica-sample-3"):
        ev = load_match(mid).events.filter(pl.col("type") == "PASS").sample(n_per_match, seed=seed)
        for f in ev["start_frame"]:
            st = frame_state(mid, int(f))
            if st.attacking_team is None or np.isnan(st.ball).any() or st.carrier is None:
                continue
            base = _sum(st)
            rec: dict[str, object] = {"match_id": mid, "frame": int(f)}
            for sigma in (0.1, 0.3):
                rhos, mad = [], []
                for _ in range(3):
                    p = st.copy()
                    on = p.on_pitch()
                    p.xy[on] += rng.normal(0.0, sigma, size=(int(on.sum()), 2))
                    p.ball = p.ball + rng.normal(0.0, sigma, size=2)
                    pert = _sum(p)
                    rhos.append(_rho(base, pert))
                    keys = set(base) & set(pert)
                    mad.append(float(np.mean([abs(base[k] - pert[k]) for k in keys])))
                rec[f"noise_{sigma}_rho"] = float(np.nanmean(rhos))
                rec[f"noise_{sigma}_mean_abs_change"] = float(np.mean(mad))
            for rf in (0.3, 0.7):
                rec[f"ring_{rf}_rho"] = _rho(base, _sum(st, ring_fraction=rf))
            rec["vs_lane_presence_rho"] = _rho(base, lane_presence(st))
            rec["mean_abs_shadow"] = float(np.mean(np.abs(list(base.values()))))
            rows.append(rec)
            print(json.dumps(rec), flush=True)
    df = pl.DataFrame(rows)
    summary = {
        c: {"mean": float(df[c].drop_nans().mean()), "median": float(df[c].drop_nans().median())}  # type: ignore[arg-type]
        for c in df.columns
        if c not in ("match_id", "frame")
    }
    result = {
        "name": "shadow_stability_v1",
        "code_revision": git_revision(),
        "n_frames": df.height,
        "runtime_s": round(time.time() - t0, 1),
        "summary": summary,
        "frames": rows,
    }
    out = EXPERIMENTS_DIR / "results" / "shadow_stability_v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
