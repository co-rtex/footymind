"""Cross-match data-quality audit: ``python -m footymind.research.data_quality_audit``.

Writes ``experiments/results/data_quality_v1/results.json`` with per-match and per-provider
counts of conditions that can bias analyses:

* frames where a team has more than 11 players with a position (substitution overlap);
* events whose start frame is missing from tracking;
* passes with no ball position at the start frame, or with no passer position there;
* passes whose end frame falls in a different period;
* SkillCorner only: the share of on-pitch positions that are detected (not extrapolated) at
  pass starts, overall and within 25 m of the ball (cross-provider distribution shift).

It only counts. It changes nothing, and nothing here is a model result.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import polars as pl

from footymind.data.store import available_matches, git_revision, load_match
from footymind.paths import EXPERIMENTS_DIR

NEAR_BALL_M = 25.0


def audit_match(mid: str) -> dict[str, Any]:
    b = load_match(mid)
    td, ev = b.tracking, b.events
    teams = np.array([p.team for p in td.players])
    on = ~np.isnan(td.xy).any(axis=2)
    frames = set(td.frame.tolist())
    passes = ev.filter(pl.col("type") == "PASS")
    ball_nan = passer_absent = cross_period = 0
    det_all: list[float] = []
    det_near: list[float] = []
    for s, e, pid in passes.select("start_frame", "end_frame", "from_player_id").iter_rows():
        if s not in frames:
            continue
        i = td.index_of_frame(int(s))
        ball = td.ball_xy[i]
        ball_nan += int(np.isnan(ball).any())
        if pid is not None and np.isnan(td.xy[i, td.player_index(pid)]).any():
            passer_absent += 1
        if e is not None and e in frames and td.period[td.index_of_frame(int(e))] != td.period[i]:
            cross_period += 1
        if td.detected is not None and on[i].any():
            det_all.append(float(td.detected[i][on[i]].mean()))
            if not np.isnan(ball).any():
                near = on[i] & (np.linalg.norm(td.xy[i] - ball[None, :], axis=1) < NEAR_BALL_M)
                if near.any():
                    det_near.append(float(td.detected[i][near].mean()))
    return {
        "provider": mid.split("-")[0],
        "n_frames": td.n_frames,
        "n_events": ev.height,
        "n_passes": passes.height,
        "frames_over_11": {
            t: int((on[:, teams == t].sum(axis=1) > 11).sum()) for t in ("home", "away")
        },
        "event_start_not_in_tracking": int(sum(f not in frames for f in ev["start_frame"])),
        "pass_ball_unobserved_at_start": ball_nan,
        "pass_passer_unobserved_at_start": passer_absent,
        "pass_ends_in_other_period": cross_period,
        "detected_share_at_pass_start": float(np.mean(det_all)) if det_all else None,
        "detected_share_near_ball_at_pass_start": float(np.mean(det_near)) if det_near else None,
    }


def main() -> dict[str, Any]:
    t0 = time.time()
    per_match = {m: audit_match(m) for m in available_matches()}
    providers: dict[str, dict[str, Any]] = {}
    for p in sorted({v["provider"] for v in per_match.values()}):
        rows = [v for v in per_match.values() if v["provider"] == p]
        agg: dict[str, Any] = {"n_matches": len(rows)}
        for k in (
            "n_frames",
            "n_events",
            "n_passes",
            "event_start_not_in_tracking",
            "pass_ball_unobserved_at_start",
            "pass_passer_unobserved_at_start",
            "pass_ends_in_other_period",
        ):
            agg[k] = int(sum(r[k] for r in rows))
        agg["frames_over_11"] = int(sum(sum(r["frames_over_11"].values()) for r in rows))
        for k in ("detected_share_at_pass_start", "detected_share_near_ball_at_pass_start"):
            vals = [r[k] for r in rows if r[k] is not None]
            agg[k] = float(np.mean(vals)) if vals else None
        providers[p] = agg
    result = {
        "name": "data_quality_v1",
        "code_revision": git_revision(),
        "providers": providers,
        "per_match": per_match,
        "runtime_s": round(time.time() - t0, 1),
    }
    d = EXPERIMENTS_DIR / "results" / "data_quality_v1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(main()["providers"], indent=2))
