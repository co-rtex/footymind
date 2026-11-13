"""Outcome validation for Defensive Shadow: does it point at the eventual interceptor?

For every attempted pass that was **intercepted** (a ``BALL LOST`` attempt followed within
2 s by a ``RECOVERY`` by the opposing team), compute at the release frame:

* Defensive Shadow (``shadow_sum``) for every outfield defender, and
* baselines using the same decision-time information:
  - nearest defender to the ball (pressure),
  - "lane presence": summed heuristic interception probability over all current pass lanes,
  - chance: 1 / number of outfield defenders.

The recovering player is the observed outcome. Metrics: top-1 and top-3 hit rates of each
ranking, with a bootstrap CI over events. No information after the release frame is used,
so there is no leakage. The pass direction is deliberately *not* used, because Defensive
Shadow is direction-agnostic.

Run with ``python -m footymind.research.shadow_validation``.
"""

from __future__ import annotations

import json
import time

import numpy as np
import polars as pl

from footymind.data.store import available_matches, git_revision, load_match
from footymind.ghost.defensive_shadow import shadow_for_state
from footymind.paths import EXPERIMENTS_DIR
from footymind.research.pass_dataset import DatasetConfig, label_events
from footymind.research.shadow_stability import lane_presence
from footymind.spatial.state import frame_state

MAX_RECOVERY_DELAY_FRAMES = 50  # 2 s at 25 Hz


def intercepted_passes(match_id: str) -> pl.DataFrame:
    ev = load_match(match_id).events.sort(["start_frame", "event_id"])
    att = label_events(ev, DatasetConfig()).filter(pl.col("type") == "BALL LOST")
    rec = ev.filter(pl.col("type") == "RECOVERY").select(
        pl.col("start_frame").alias("rec_frame"),
        pl.col("team").alias("rec_team"),
        pl.col("from_player_id").alias("interceptor"),
    )
    out = []
    for r in att.iter_rows(named=True):
        f = int(r["start_frame"])
        nxt = rec.filter(
            (pl.col("rec_frame") >= f)
            & (pl.col("rec_frame") <= f + MAX_RECOVERY_DELAY_FRAMES)
            & (pl.col("rec_team") != r["team"])
        ).head(1)
        if nxt.height and nxt["interceptor"][0] is not None:
            out.append(
                {
                    "match_id": match_id,
                    "frame": f,
                    "team": r["team"],
                    "interceptor": nxt["interceptor"][0],
                }
            )
    return pl.DataFrame(out)


def _rank_hits(scores: dict[str, float], target: str) -> tuple[int, int]:
    order = sorted(scores, key=lambda k: -scores[k])
    return int(order[:1] == [target]), int(target in order[:3])


def main(seed: int = 0) -> None:
    t0 = time.time()
    rows = []
    for mid in available_matches():
        for r in intercepted_passes(mid).iter_rows(named=True):
            st = frame_state(mid, r["frame"])
            if st.attacking_team != r["team"] or np.isnan(st.ball).any():
                continue
            try:
                sh = shadow_for_state(st)
            except ValueError:
                continue
            shadow = {d.player_id: d.shadow_sum for d in sh.defenders}
            if r["interceptor"] not in shadow:
                continue  # interceptor was the goalkeeper (excluded from shadow)
            outfield = list(shadow)
            idx = [st.index(p) for p in outfield]
            dist = np.linalg.norm(st.xy[idx] - st.ball[None, :], axis=1)
            nearest = {p: -float(d) for p, d in zip(outfield, dist, strict=True)}
            lp = {p: v for p, v in lane_presence(st).items() if p in shadow}
            rec: dict[str, object] = {**r, "n_defenders": len(outfield)}
            for name, sc in (
                ("shadow", shadow),
                ("nearest_to_ball", nearest),
                ("lane_presence", lp),
            ):
                h1, h3 = _rank_hits(sc, r["interceptor"])
                rec[f"{name}_top1"], rec[f"{name}_top3"] = h1, h3
            rows.append(rec)
    df = pl.DataFrame(rows)
    rng = np.random.default_rng(seed)
    summary: dict[str, object] = {"n_events": df.height}
    n_def = df["n_defenders"].to_numpy()
    summary["chance_top1"] = float(np.mean(1 / n_def))
    summary["chance_top3"] = float(np.mean(np.minimum(3 / n_def, 1)))
    for c in [c for c in df.columns if c.endswith(("_top1", "_top3"))]:
        v = df[c].to_numpy().astype(float)
        boots = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(2000)]
        summary[c] = {
            "rate": float(v.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        }
    # Paired difference: shadow vs lane_presence (top-3).
    d = (df["shadow_top3"] - df["lane_presence_top3"]).to_numpy().astype(float)
    bd = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(2000)]
    summary["shadow_minus_lane_presence_top3"] = {
        "delta": float(d.mean()),
        "ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
    }
    d2 = (df["shadow_top3"] - df["nearest_to_ball_top3"]).to_numpy().astype(float)
    bd2 = [d2[rng.integers(0, d2.size, d2.size)].mean() for _ in range(2000)]
    summary["shadow_minus_nearest_top3"] = {
        "delta": float(d2.mean()),
        "ci95": [float(np.percentile(bd2, 2.5)), float(np.percentile(bd2, 97.5))],
    }
    result = {
        "name": "shadow_validation_v1",
        "code_revision": git_revision(),
        "runtime_s": round(time.time() - t0, 1),
        "summary": summary,
        "events": rows,
    }
    out = EXPERIMENTS_DIR / "results" / "shadow_validation_v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
