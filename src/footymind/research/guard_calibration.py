"""How far before the event start frame is the situation pre-kick? (ADR-0007 calibration)

``python -m footymind.research.guard_calibration`` writes
``experiments/results/guard_calibration_v1/results.json``.

For every PASS event, at several offsets before its start frame, it counts how often the
ball is moving away from the passer faster than ``AWAY_MPS``. That is measured as a
one-frame velocity projected on the passer→ball direction. At the true release this share
is high. Well before the release it settles at a provider-specific floor: dribble touches,
plus ball-position noise, which at 10 fps is about 3 m/s per 0.3 m of jitter. A guard is
adequate once the share has reached that floor.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import polars as pl

from footymind.data.store import available_matches, git_revision, load_match
from footymind.paths import EXPERIMENTS_DIR

AWAY_MPS = 3.0
OFFSETS_S = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6)


def kick_underway_share(match_ids: list[str], offset_s: float) -> tuple[float, int]:
    hit = tot = 0
    for mid in match_ids:
        b = load_match(mid)
        td = b.tracking
        k = round(offset_s * td.fps)
        passes = b.events.filter(pl.col("type") == "PASS").select("start_frame", "from_player_id")
        for s, pid in passes.iter_rows():
            if pid is None:
                continue
            i = td.index_of_frame(int(s)) - k
            if i < 1 or td.period[i - 1] != td.period[i]:
                continue
            p, b1, b0 = td.xy[i, td.player_index(pid)], td.ball_xy[i], td.ball_xy[i - 1]
            if np.isnan(p).any() or np.isnan(b1).any() or np.isnan(b0).any():
                continue
            away = b1 - p
            n = float(np.linalg.norm(away))
            if n < 1e-6:
                continue
            tot += 1
            hit += int(float(((b1 - b0) * td.fps) @ (away / n)) > AWAY_MPS)
    return hit / max(tot, 1), tot


def guard_ball_fallback(match_ids: list[str], guard_s: float = 0.2) -> dict[str, Any]:
    """How often the guarded dataset falls back to the start-frame ball (session-4 review).

    ``pass_dataset._features_for`` uses the start-frame ball when the guard-frame ball is
    unobserved. That is harmless for restarts (the ball is placed at the start frame), and
    for open play it uses the start-frame ball, which is at the passer's feet in almost all
    Metrica passes (C38). This counts both cases.
    """
    from footymind.research.pass_dataset import DatasetConfig, _prev_event_features, label_events

    n = nan = restart = 0
    for mid in match_ids:
        b = load_match(mid)
        td = b.tracking
        k = round(guard_s * td.fps)
        ev = label_events(_prev_event_features(b.events), DatasetConfig())
        for r in ev.select("start_frame", "last_set_piece_frame").iter_rows():
            f0, lsp = int(r[0]), r[1]
            i0 = td.index_of_frame(f0)
            i = i0 - k
            if i < 0 or td.period[i] != td.period[i0]:
                continue
            n += 1
            if np.isnan(td.ball_xy[i]).any():
                nan += 1
                restart += int(lsp is not None and 0 <= f0 - int(lsp) <= k)
    return {
        "pass_rows": n,
        "ball_unobserved_at_guard": nan,
        "of_which_restarts": restart,
        "open_play_fallbacks": nan - restart,
    }


def main() -> dict[str, Any]:
    t0 = time.time()
    out: dict[str, Any] = {"name": "guard_calibration_v1", "code_revision": git_revision()}
    out["away_mps"] = AWAY_MPS
    providers: dict[str, Any] = {}
    for prov in ("metrica", "skillcorner"):
        mids = [m for m in available_matches() if m.startswith(prov + "-")]
        if not mids:
            continue
        rows = []
        for off in OFFSETS_S:
            share, n = kick_underway_share(mids, off)
            rows.append({"offset_s": off, "share_ball_moving_away": share, "n": n})
        providers[prov] = {
            "n_matches": len(mids),
            "by_offset": rows,
            "guard_ball_fallback": guard_ball_fallback(mids),
        }
    out["providers"] = providers
    out["runtime_s"] = round(time.time() - t0, 1)
    d = EXPERIMENTS_DIR / "results" / "guard_calibration_v1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    r = main()
    for p, v in r["providers"].items():
        print(p, [(x["offset_s"], round(x["share_ball_moving_away"], 3)) for x in v["by_offset"]])
