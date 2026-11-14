"""Attack vs defense movement decomposition of passing opportunity (pre-registered, session 3).

``python -m footymind.research.movement_decomposition_study <provider> [n_per_match]``
writes ``experiments/results/movement_decomposition_v1[_<provider>]/results.json``.

Samples: attempted passes (completed and failed), ``n_per_match`` per match (seeded), at the
release frame. The window is ``HORIZON_S`` = 1.0 s within the same period. See
``ghost/movement_decomposition.py`` for the definitions.

Hypotheses and decision rules (fixed before running):

Primary measure: the best *progressive* option (see ``ghost/movement_decomposition.py``).

* **H-G1 (cat and mouse).** In the second before a pass, off-ball attacking movement raises
  the team's best progressive option (mean attack effect > 0, CI excludes 0) and defensive movement
  lowers it (mean defense effect < 0, CI excludes 0).
* **H-G2 (identifiability).** The interaction is not negligible: mean |interaction| is at
  least 25% of mean |attack effect| (lower CI bound ≥ 0.25). If so, the attacker credit from
  holding defenders at their observed positions (``run_value``) is not well defined without
  a response model.
* Descriptive: correlation between attack and defense effects across samples. On
  SkillCorner, a sensitivity subset where ≥ 80% of the earlier positions within 25 m of the
  ball are detected rather than extrapolated.

Uncertainty: bootstrap over (match, possession) clusters.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
import polars as pl

from footymind.data.store import available_matches, git_revision, load_match
from footymind.ghost.movement_decomposition import decompose
from footymind.paths import EXPERIMENTS_DIR
from footymind.research.pass_dataset import DatasetConfig, _prev_event_features, label_events
from footymind.spatial.state import causal_velocities, frame_state

HORIZON_S = 1.0
NEAR_BALL_M = 25.0
DETECTED_SUBSET_MIN = 0.8


def _samples(provider: str, n_per_match: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mid in [m for m in available_matches() if m.startswith(provider + "-")]:
        b = load_match(mid)
        td = b.tracking
        vel = causal_velocities(mid)
        ev = label_events(_prev_event_features(b.events), DatasetConfig())
        ev = ev.sample(min(n_per_match, ev.height), seed=seed)
        k = round(HORIZON_S * td.fps)
        for r in ev.iter_rows(named=True):
            f = int(r["start_frame"])
            i = td.index_of_frame(f)
            if i - k < 0 or td.period[i - k] != td.period[i]:
                continue
            st = frame_state(mid, f)
            if st.attacking_team != r["team"] or st.carrier is None or np.isnan(st.ball).any():
                continue
            earlier_xy = td.xy[i - k].astype(np.float64)
            try:
                d = decompose(st, earlier_xy, vel[i - k].astype(np.float64))
            except ValueError:
                continue
            det = np.nan
            if td.detected is not None:
                near = st.on_pitch() & (
                    np.linalg.norm(st.xy - st.ball[None, :], axis=1) < NEAR_BALL_M
                )
                det = float(td.detected[i - k][near].mean()) if near.any() else np.nan
            row: dict[str, Any] = {
                "match_id": mid,
                "possession_id": r["possession_id"],
                "frame": f,
                "success": int(r["success"]),
                "earlier_detected_near_ball": det,
            }
            for m, e in d.effects.items():
                sfx = "" if m == "forward" else f"_{m}"
                row.update(
                    {
                        f"attack{sfx}": e.attack,
                        f"defense{sfx}": e.defense,
                        f"interaction{sfx}": e.interaction,
                    }
                )
            row.update(v11=d.v["forward"]["11"], v00=d.v["forward"]["00"])
            rows.append(row)
    return rows


def _stats(df: pl.DataFrame, n_boot: int, seed: int, suffix: str = "") -> dict[str, Any]:
    a = df[f"attack{suffix}"].to_numpy()
    d = df[f"defense{suffix}"].to_numpy()
    it = df[f"interaction{suffix}"].to_numpy()
    groups = (df["match_id"] + "|" + df["possession_id"].fill_null("none")).to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    n_groups = int(uniq.size)
    members = [np.flatnonzero(inv == g) for g in range(n_groups)]

    def point(idx: np.ndarray) -> dict[str, float]:
        aa, dd, ii = a[idx], d[idx], it[idx]
        return {
            "mean_attack": float(aa.mean()),
            "mean_defense": float(dd.mean()),
            "mean_interaction": float(ii.mean()),
            "mean_abs_interaction_over_mean_abs_attack": float(
                np.abs(ii).mean() / max(np.abs(aa).mean(), 1e-12)
            ),
            "corr_attack_defense": float(np.corrcoef(aa, dd)[0, 1]),
        }

    full = point(np.arange(a.size))
    rng = np.random.default_rng(seed)
    boots: dict[str, list[float]] = {k: [] for k in full}
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        for k, v in point(idx).items():
            boots[k].append(v)
    return {
        "n": int(a.size),
        "n_clusters": len(members),
        **{
            k: {
                "value": v,
                "ci95": [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))],
            }
            for k, v in full.items()
        },
    }


def main(
    provider: str = "metrica", n_per_match: int = 150, seed: int = 0, n_boot: int = 1000
) -> dict[str, Any]:
    t0 = time.time()
    df = pl.DataFrame(_samples(provider, n_per_match, seed))
    primary = _stats(df, n_boot, seed)
    result: dict[str, Any] = {
        "name": "movement_decomposition_v1" + ("" if provider == "metrica" else f"_{provider}"),
        "provider": provider,
        "code_revision": git_revision(),
        "horizon_s": HORIZON_S,
        "n_per_match": n_per_match,
        "measure_primary": "forward (best progressive option, > 5 m ahead of the ball)",
        "forward": primary,
        "sum": _stats(df, n_boot, seed, "_sum"),
        "any": _stats(df, n_boot, seed, "_any"),
        "hypotheses": {
            "H-G1": bool(
                primary["mean_attack"]["ci95"][0] > 0 and primary["mean_defense"]["ci95"][1] < 0
            ),
            "H-G2": bool(primary["mean_abs_interaction_over_mean_abs_attack"]["ci95"][0] >= 0.25),
        },
    }
    if provider != "metrica":
        sub = df.filter(pl.col("earlier_detected_near_ball") >= DETECTED_SUBSET_MIN)
        result["detected_subset"] = {
            "rule": f"earlier positions within {NEAR_BALL_M} m of the ball >= "
            f"{DETECTED_SUBSET_MIN:.0%} detected",
            "forward": _stats(sub, n_boot, seed) if sub.height > 30 else None,
        }
    result["runtime_s"] = round(time.time() - t0, 1)
    d = EXPERIMENTS_DIR / "results" / result["name"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    r = main(
        sys.argv[1] if len(sys.argv) > 1 else "metrica",
        int(sys.argv[2]) if len(sys.argv) > 2 else 150,
    )
    print(json.dumps({k: r[k] for k in ("hypotheses", "forward")}, indent=2))
