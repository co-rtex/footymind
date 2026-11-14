"""Measure server-side API latency (p50/p95) for analysis, scenario and similar moments.

    uv run python scripts/api_latency.py            # writes experiments/results/api_latency_v1
    uv run python scripts/api_latency.py --dry-run  # print, do not write

In-process (FastAPI TestClient, no network), one request at a time, on real processed data
(FOOTYMIND_DATA_DIR) and the on-disk similar-moments indexes. Frames are a seeded sample of
pass start frames from two matches per provider. Each match is warmed up once first (bundle
load, model load and index load are reported separately as "cold"), so the warm numbers are the
steady state a user sees while browsing. Requests use the web lab's defaults: analysis with the
pitch-control grid, scenarios with the default 24-member ensemble (policy none and
marker_follows), similar moments with k = 8. Latency depends on the machine and its load; the
results record both. Refuses to write from a dirty src/ tree (the code revision is recorded).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Any

import numpy as np
import polars as pl

from footymind.data.store import available_matches, git_revision, load_match
from footymind.paths import REPO_ROOT

NAME = "api_latency_v1"
SEED = 20261114
N_FRAMES = {"analysis": 40, "scenario": 12, "similar": 40}


def _frames(match_id: str, n: int, rng: np.random.Generator) -> list[int]:
    ev = load_match(match_id).events.filter(pl.col("type") == "PASS")
    idx = rng.choice(ev.height, size=min(n, ev.height), replace=False)
    return sorted(int(ev["start_frame"][int(i)]) for i in idx)


def _stats(ms: list[float]) -> dict[str, float | int]:
    a = np.asarray(ms)
    return {
        "n": int(a.size),
        "p50_ms": round(float(np.percentile(a, 50)), 1),
        "p95_ms": round(float(np.percentile(a, 95)), 1),
        "max_ms": round(float(a.max()), 1),
        "mean_ms": round(float(a.mean()), 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    rev = git_revision()
    if rev.endswith("+dirty") and not args.dry_run:
        print("refusing: src/ has uncommitted changes", file=sys.stderr)
        return 1

    from fastapi.testclient import TestClient

    from footymind.api.app import app

    c = TestClient(app)
    rng = np.random.default_rng(SEED)
    out: dict[str, Any] = {
        "name": NAME,
        "code_revision": rev,
        "method": __doc__.strip().split("\n\n", 2)[2].replace("\n", " "),
        "seed": SEED,
        "machine": {
            "cpu_count": os.cpu_count(),
            "load_avg_1m_at_start": round(os.getloadavg()[0], 2),
            "python": platform.python_version(),
            "platform": platform.platform(terse=True),
        },
        "providers": {},
    }
    errors: list[str] = []

    def timed(method: str, path: str, body: Any = None) -> float:
        t0 = time.perf_counter()
        r = c.post(path, json=body) if method == "POST" else c.get(path)
        dt = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            errors.append(f"{r.status_code} {path}")
        return dt

    for provider in ("metrica", "skillcorner"):
        mids = [m for m in available_matches() if m.startswith(provider + "-")][:2]
        if not mids:
            continue
        cold: dict[str, float] = {}
        ms: dict[str, list[float]] = {
            "analysis": [],
            "scenario_none": [],
            "scenario_marker_follows": [],
            "similar": [],
        }
        for mid in mids:
            frames = _frames(mid, max(N_FRAMES.values()), rng)
            f0 = frames[0]
            cold[f"analysis_first_call_{mid}"] = round(
                timed("GET", f"/api/matches/{mid}/frames/{f0}/analysis"), 1
            )
            if "similar_first_call" not in cold:
                cold["similar_first_call"] = round(
                    timed("GET", f"/api/matches/{mid}/frames/{f0}/similar"), 1
                )
            for f in frames[: N_FRAMES["analysis"] // len(mids)]:
                ms["analysis"].append(timed("GET", f"/api/matches/{mid}/frames/{f}/analysis"))
            for f in frames[: N_FRAMES["similar"] // len(mids)]:
                t0 = time.perf_counter()
                r = c.get(f"/api/matches/{mid}/frames/{f}/similar?k=8")
                if r.status_code == 200:  # 404: no labelled decision at this frame (excluded)
                    ms["similar"].append((time.perf_counter() - t0) * 1000)
                elif r.status_code != 404:
                    errors.append(f"{r.status_code} similar {mid}@{f}")
            for f in frames[: N_FRAMES["scenario"] // len(mids)]:
                a = c.get(f"/api/matches/{mid}/frames/{f}/analysis?grid=false").json()
                mover = next(
                    (
                        p
                        for p in a["players"]
                        if p["team"] == a["attacking_team"]
                        and p["player_id"] != a["carrier_id"]
                        and p["x"] is not None
                    ),
                    None,
                )
                if mover is None:
                    continue
                for policy in ("none", "marker_follows"):
                    body = {
                        "match_id": mid,
                        "frame": f,
                        "interventions": [
                            {
                                "player_id": mover["player_id"],
                                "x": max(-50.0, min(50.0, mover["x"] + 3.0)),
                                "y": mover["y"],
                            }
                        ],
                        "response_policy": policy,
                    }
                    ms[f"scenario_{policy}"].append(timed("POST", "/api/scenarios", body))
        out["providers"][provider] = {
            "matches": mids,
            "warm": {k: _stats(v) for k, v in ms.items() if v},
            "cold_ms": cold,
        }
    out["errors"] = errors
    text = json.dumps(out, indent=2)
    print(text)
    if errors:
        print(f"{len(errors)} unexpected responses", file=sys.stderr)
        return 1
    if not args.dry_run:
        d = REPO_ROOT / "experiments" / "results" / NAME
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text(text + "\n")
        print(f"wrote {d / 'results.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
