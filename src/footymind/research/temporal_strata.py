"""Stratified (descriptive) analysis of a temporal experiment: where does temporal context help?

``python -m footymind.research.temporal_strata <experiment_name>`` reads the experiment's
out-of-fold predictions, rebuilds its dataset (deterministic) to recover stratum columns,
and re-runs the pre-specified comparisons inside each stratum. It writes
``experiments/results/<name>_strata/results.json``.

Strata (pre-registered in docs/RESEARCH.md, RQ2):

* ``tracking_quality``: share of detected positions in the 1 s window, split at the
  provider median. It is constant (1.0) for optical tracking, so it is skipped there.
* ``situation``: restart vs open play. This replaced a quick-vs-slower release split after
  the independent review (2026-11-14): ``quick_release`` is itself a model feature, so gains
  inside its strata are partly mechanical. ``is_restart`` is in every model.

These are descriptive. Strata are not used to select models, and no stratum result is a
headline claim on its own.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
import polars as pl

from footymind.data.store import git_revision
from footymind.paths import EXPERIMENTS_DIR
from footymind.research.evaluate import match_level_bootstrap, metrics, paired_bootstrap
from footymind.research.pass_dataset import DatasetConfig, build

STRATIFIED_COMPARISONS = [
    ("Adt_logreg", "Adt_tob_logreg"),
    ("Bdt_tob_logreg", "Bdt_tt_logreg"),
    ("Adt_tob_logreg", "Bdt_tob_logreg"),
]


def strata_masks(df: pl.DataFrame) -> dict[str, pl.Series]:
    """Boolean masks keyed by ``<stratum>=<level>`` (descriptive strata only)."""
    out: dict[str, pl.Series] = {}
    q = df["history_detected_frac"]
    if q.n_unique() > 1:
        med = float(q.median())  # type: ignore[arg-type]
        out["tracking_quality=low"] = q <= med
        out["tracking_quality=high"] = q > med
    out["situation=open_play"] = df["is_restart"] == 0
    out["situation=restart"] = df["is_restart"] == 1
    return out


def match_level_only(name: str, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Two-stage (match, possession) bootstrap from the out-of-fold predictions only.

    It needs no dataset rebuild. It writes ``<name>_match_level/results.json``.
    """
    res_dir = EXPERIMENTS_DIR / "results" / name
    src = json.loads((res_dir / "results.json").read_text())
    oof = pl.read_parquet(res_dir / "oof_predictions.parquet")
    pairs = [*STRATIFIED_COMPARISONS, ("Adt_logreg", "Bdt_logreg")]
    result = {
        "name": f"{name}_match_level",
        "source_experiment": name,
        "source_code_revision": src.get("code_revision"),
        "code_revision": git_revision(),
        "match_level": [
            match_level_bootstrap(oof, a, b, n_boot, seed)
            for a, b in pairs
            if f"p_{a}" in oof.columns and f"p_{b}" in oof.columns
        ],
    }
    d = EXPERIMENTS_DIR / "results" / f"{name}_match_level"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(name: str, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    t0 = time.time()
    res_dir = EXPERIMENTS_DIR / "results" / name
    cfg = json.loads((res_dir / "results.json").read_text())["config"]
    oof = pl.read_parquet(res_dir / "oof_predictions.parquet")
    ds = build(cfg["matches"], DatasetConfig(**cfg.get("dataset", {})))
    meta = ds.select("match_id", "event_id", "history_detected_frac", "is_restart")
    joined = oof.join(meta, on=["match_id", "event_id"], how="left")
    if joined["is_restart"].null_count():
        raise ValueError("could not recover stratum columns for every out-of-fold row")
    out: dict[str, Any] = {}
    for key, mask in strata_masks(joined).items():
        sub = joined.filter(mask)
        y = sub["success"].to_numpy()
        out[key] = {
            "n": sub.height,
            "success_rate": float(np.mean(y)),
            "metrics": {
                m: metrics(y, sub[f"p_{m}"].to_numpy())
                for pair in STRATIFIED_COMPARISONS
                for m in pair
            },
            "comparisons": [
                paired_bootstrap(sub, a, b, n_boot, seed) for a, b in STRATIFIED_COMPARISONS
            ],
        }
    # Between-match uncertainty for the primary comparisons and for the C20/C28 contrast.
    match_level = [
        match_level_bootstrap(oof, a, b, n_boot, seed)
        for a, b in [*STRATIFIED_COMPARISONS, ("Adt_logreg", "Bdt_logreg")]
    ]
    result = {
        "name": f"{name}_strata",
        "match_level": match_level,
        "source_experiment": name,
        "source_code_revision": json.loads((res_dir / "results.json").read_text()).get(
            "code_revision"
        ),
        "code_revision": git_revision(),
        "tracking_quality_median": float(joined["history_detected_frac"].median()),  # type: ignore[arg-type]
        "strata": out,
        "runtime_s": round(time.time() - t0, 1),
        "note": "Descriptive strata; not used for model selection.",
    }
    d = EXPERIMENTS_DIR / "results" / f"{name}_strata"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--match-level-only":
        for c in match_level_only(sys.argv[1])["match_level"]:
            print(c["a"], c["b"], c["log_loss"], c["n_matches_b_better"], "/", c["n_matches"])
        raise SystemExit(0)
    r = main(sys.argv[1])
    for k, v in r["strata"].items():
        print(k, v["n"], [(c["a"], c["b"], c["log_loss"]) for c in v["comparisons"]])
    for c in r["match_level"]:
        print(
            "match-level",
            c["a"],
            c["b"],
            c["log_loss"],
            c["n_matches_b_better"],
            "/",
            c["n_matches"],
        )
