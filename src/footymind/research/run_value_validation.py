"""Outcome validation for run value: do players whose movement made them available get the ball?

Pre-specified test (written before any result was seen):
* Sample completed passes (seeded; ``n_per_match`` per match, all 3 matches).
* At the pass start frame, compute every non-carrier attacker's own run value
  (``ghost.run_value``, horizon 1 s) and current option probability.
* Outcome: the actual receiver (known for completed passes).
* Rankings compared by top-1 and top-3 hit rate: run value, current option probability
  (availability without movement information), current speed, and chance.
* Primary statistic: mean run value of the chosen receiver minus the mean over non-chosen
  teammates, with a bootstrap CI over passes. A positive value means that players whose last
  second of movement increased their availability are chosen more often.
* Secondary statistic: does adding run value to availability help? Compare the top-1 hit rate
  of availability against availability + run value (z-scored sum), paired bootstrap.

No information after the pass start frame is used.
Run with ``python -m footymind.research.run_value_validation``.
"""

from __future__ import annotations

import json
import time

import numpy as np
import polars as pl

from footymind.data.store import available_matches, git_revision, load_match
from footymind.ghost.run_value import run_values
from footymind.paths import EXPERIMENTS_DIR


def _top(scores: dict[str, float], target: str, k: int) -> int:
    return int(target in sorted(scores, key=lambda p: -scores[p])[:k])


def _z(d: dict[str, float]) -> dict[str, float]:
    v = np.array(list(d.values()))
    s = v.std() or 1.0
    return {k: (x - v.mean()) / s for k, x in d.items()}


def main(provider: str = "metrica", n_per_match: int = 150, seed: int = 0) -> None:
    """Run per provider (never pooled). Metrica writes ``run_value_validation_v1``."""
    t0 = time.time()
    rows = []
    for mid in [m for m in available_matches() if m.startswith(provider + "-")]:
        ev = load_match(mid).events.filter(
            (pl.col("type") == "PASS") & pl.col("to_player_id").is_not_null()
        )
        ev = ev.sample(min(n_per_match, ev.height), seed=seed)
        for r in ev.iter_rows(named=True):
            try:
                rv = run_values(mid, int(r["start_frame"]))
            except ValueError:
                continue
            ids = [x.player_id for x in rv]
            if r["to_player_id"] not in ids or len(ids) < 3:
                continue
            run = {x.player_id: x.own_run_value for x in rv}
            avail = {x.player_id: x.p_observed for x in rv}
            speed = {x.player_id: (x.speed_mps or 0.0) for x in rv}
            combo = {k: _z(avail)[k] + _z(run)[k] for k in ids}
            target = r["to_player_id"]
            others = [run[k] for k in ids if k != target]
            rec: dict[str, object] = {
                "match_id": mid,
                "frame": int(r["start_frame"]),
                "receiver": target,
                "n_candidates": len(ids),
                "run_receiver": run[target],
                "run_others_mean": float(np.mean(others)),
            }
            for name, sc in (
                ("run", run),
                ("availability", avail),
                ("speed", speed),
                ("availability_plus_run", combo),
            ):
                rec[f"{name}_top1"] = _top(sc, target, 1)
                rec[f"{name}_top3"] = _top(sc, target, 3)
            rows.append(rec)
    df = pl.DataFrame(rows)
    rng = np.random.default_rng(seed)

    def boot(v: np.ndarray) -> list[float]:
        b = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(2000)]
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    n = df["n_candidates"].to_numpy()
    summary: dict[str, object] = {
        "n_passes": df.height,
        "chance_top1": float(np.mean(1 / n)),
        "chance_top3": float(np.mean(np.minimum(3 / n, 1))),
    }
    diff = (df["run_receiver"] - df["run_others_mean"]).to_numpy()
    summary["run_value_receiver_minus_others"] = {"mean": float(diff.mean()), "ci95": boot(diff)}
    for c in [c for c in df.columns if c.endswith(("_top1", "_top3"))]:
        v = df[c].to_numpy().astype(float)
        summary[c] = {"rate": float(v.mean()), "ci95": boot(v)}
    d2 = (df["availability_plus_run_top1"] - df["availability_top1"]).to_numpy().astype(float)
    summary["combo_minus_availability_top1"] = {"delta": float(d2.mean()), "ci95": boot(d2)}
    name = "run_value_validation_v1" + ("" if provider == "metrica" else f"_{provider}")
    result = {
        "name": name,
        "provider": provider,
        "code_revision": git_revision(),
        "runtime_s": round(time.time() - t0, 1),
        "summary": summary,
        "passes": rows,
    }
    out = EXPERIMENTS_DIR / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "metrica")
