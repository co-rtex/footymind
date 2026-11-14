"""How strongly does each response policy react to an intervention? (descriptive probe)

The forecast validation (``response_validation``) says nothing about the *size* of a
policy's response to a hypothetical change. This probe measures it on real frames. For
pass-start frames with an identified carrier, it relocates the carrier (or, for
``lane_cover``, the receiver of the pass) by ``d`` metres in four directions and records the
simulated defender moves of each policy at the default scenario horizon (1.5 s).

Reported per policy and displacement: the share of probes with any simulated move, the mean
number of defenders moved, the mean and 95th percentile of the largest move, and the mean
total displacement summed over defenders. The output is descriptive (no claim of
correctness, since there is no ground truth for responses). It exists so the size of the
simulated responses is stated, not guessed.

Run with ``python -m footymind.research.response_sensitivity <provider> [max_frames]``.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import polars as pl

from footymind.counterfactual.response import Intervention, ResponseContext, apply_response
from footymind.counterfactual.response_model import load_response_model
from footymind.counterfactual.scenario import ball_velocity
from footymind.data.store import available_matches, git_revision, load_match
from footymind.paths import EXPERIMENTS_DIR
from footymind.spatial.state import frame_state

POLICIES = ("marker_follows", "press_carrier", "zonal_shift", "lane_cover", "learned")
DISPLACEMENTS_M = (1.0, 5.0)
DIRECTIONS = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
HORIZON_S = 1.5  # scenario default


def main(provider: str = "metrica", max_frames: int = 300, seed: int = 0) -> dict[str, Any]:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    mids = [m for m in available_matches() if m.startswith(provider + "-")]
    model = load_response_model(provider)
    cands = []
    for mid in mids:
        ev = load_match(mid).events.filter(
            (pl.col("type") == "PASS") & pl.col("to_player_id").is_not_null()
        )
        pairs = zip(ev["start_frame"], ev["to_player_id"], strict=True)
        cands += [(mid, int(f), str(t)) for f, t in pairs]
    order = rng.permutation(len(cands))
    acc: dict[str, dict[str, list[tuple[int, float, float]]]] = {
        p: {f"{d:g}m": [] for d in DISPLACEMENTS_M} for p in POLICIES
    }
    used = 0
    for k in order:
        if used >= max_frames:
            break
        mid, f, receiver = cands[k]
        s = frame_state(mid, f)
        if s.carrier is None or s.attacking_team is None or receiver not in s.player_ids:
            continue
        r = s.index(receiver)
        if np.isnan(s.xy[r]).any() or r == s.carrier:
            continue
        used += 1
        ctx = ResponseContext(
            pitch=load_match(mid).tracking.pitch,
            horizon_s=HORIZON_S,
            ball_velocity=ball_velocity(mid, f),
            model=model,
        )
        for pol in POLICIES:
            if pol == "learned" and model is None:
                continue
            who = r if pol in ("marker_follows", "lane_cover") else s.carrier
            for d in DISPLACEMENTS_M:
                for ux, uy in DIRECTIONS:
                    hyp = s.copy()
                    new = s.xy[who] + d * np.array([ux, uy])
                    hyp.xy[who] = new
                    if who == s.carrier:
                        hyp.ball = hyp.ball + d * np.array([ux, uy])
                    iv = [Intervention(s.player_ids[who], float(new[0]), float(new[1]))]
                    moves, _ = apply_response(s, hyp, iv, pol, ctx)
                    dist = [
                        float(np.hypot(m.to_xy[0] - m.from_xy[0], m.to_xy[1] - m.from_xy[1]))
                        for m in moves
                    ]
                    acc[pol][f"{d:g}m"].append((len(dist), max(dist, default=0.0), sum(dist)))
    summary: dict[str, Any] = {}
    for pol, by_d in acc.items():
        summary[pol] = {}
        for dk, vals in by_d.items():
            if not vals:
                continue
            a = np.array(vals)
            summary[pol][dk] = {
                "n_probes": len(vals),
                "share_any_move": float(np.mean(a[:, 0] > 0)),
                "mean_defenders_moved": float(a[:, 0].mean()),
                "mean_max_move_m": float(a[:, 1].mean()),
                "p95_max_move_m": float(np.percentile(a[:, 1], 95)),
                "mean_total_move_m": float(a[:, 2].mean()),
            }
    name = "response_sensitivity_v1" + ("" if provider == "metrica" else f"_{provider}")
    result = {
        "name": name,
        "provider": provider,
        "matches": mids,
        "code_revision": git_revision(),
        "response_model_code_revision": model.meta.get("code_revision") if model else None,
        "runtime_s": round(time.time() - t0, 1),
        "n_frames": used,
        "horizon_s": HORIZON_S,
        "displacements_m": list(DISPLACEMENTS_M),
        "note": "descriptive: size of simulated responses, not their correctness",
        "summary": summary,
    }
    out = EXPERIMENTS_DIR / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return result


if __name__ == "__main__":
    import sys

    main(
        sys.argv[1] if len(sys.argv) > 1 else "metrica",
        max_frames=int(sys.argv[2]) if len(sys.argv) > 2 else 300,
    )
