"""Reproducible experiment runner: ``footymind experiment experiments/configs/<name>.json``.

Writes ``experiments/results/<name>/`` containing:

* ``results.json``: config, code revision, input data hashes, dataset summary,
  per-fold and pooled metrics, bootstrap comparisons, calibration and
  importances. It is committed.
* ``report.md``: a human-readable table generated from ``results.json``. It is committed.
* ``oof_predictions.parquet``: out-of-fold predictions. It is not committed (see .gitignore).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from footymind.data.store import git_revision, load_match
from footymind.paths import EXPERIMENTS_DIR
from footymind.research.evaluate import (
    calibration_curve,
    lomo_predictions,
    metrics,
    model_specs,
    paired_bootstrap,
    permutation_importances,
)
from footymind.research.pass_dataset import DatasetConfig, build, inferred_receiver_accuracy


def run_from_config(path: str | Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text())
    t0 = time.time()
    name = cfg["name"]
    seed = int(cfg.get("seed", 0))
    ds_cfg = DatasetConfig(**cfg.get("dataset", {}))
    df = build(cfg["matches"], ds_cfg)
    for flt in cfg.get("filters", []):
        if flt == "exclude_truncated_release":
            df = df.filter(pl.col("release_truncated") == 0)
        elif flt == "exclude_after_own_carry":
            df = df.filter(pl.col("after_own_carry") == 0)
        else:
            raise ValueError(f"unknown filter {flt}")
    specs = model_specs()
    if "models" in cfg:
        specs = {k: v for k, v in specs.items() if k in cfg["models"]}
    oof, fold_metrics, fitted = lomo_predictions(df, specs, seed)
    y = oof["success"].to_numpy()
    pooled = {n: metrics(y, oof[f"p_{n}"].to_numpy()) for n in specs}
    comparisons = [
        paired_bootstrap(oof, a, b, int(cfg.get("n_boot", 2000)), seed)
        for a, b in cfg.get("comparisons", [])
    ]
    calib = {n: calibration_curve(y, oof[f"p_{n}"].to_numpy()) for n in cfg.get("calibration", [])}
    importances = {
        n: permutation_importances(fitted, specs[n], seed) for n in cfg.get("importances", [])
    }
    hashes = {m: load_match(m).manifest["provenance"]["files"] for m in cfg["matches"]}
    summary = {
        "n": df.height,
        "success_rate": float(y.mean()),
        "by_match": {
            m: {
                "n": int((df["match_id"] == m).sum()),
                "success_rate": float(y[(df["match_id"] == m).to_numpy()].mean()),
            }
            for m in cfg["matches"]
        },
        "by_label_source": dict(df.group_by("label_source").len().sort("label_source").iter_rows()),
        "n_truncated_release": int(df["release_truncated"].sum()),
        "receiver_inference": inferred_receiver_accuracy(df),
    }
    result = {
        "name": name,
        "config": cfg,
        "code_revision": git_revision(),
        "data_files_sha256": hashes,
        "runtime_s": round(time.time() - t0, 1),
        "dataset": summary,
        "models": {
            n: {"family": s.family, "description": s.description, "n_features": len(s.features)}
            for n, s in specs.items()
        },
        "pooled_oof_metrics": pooled,
        "fold_metrics": fold_metrics,
        "comparisons": comparisons,
        "calibration": calib,
        "permutation_importance": importances,
    }
    out = EXPERIMENTS_DIR / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2, default=_json_default))
    oof.write_parquet(out / "oof_predictions.parquet")
    (out / "report.md").write_text(render_report(result))
    print((out / "report.md").read_text())
    return result


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def render_report(r: dict[str, Any]) -> str:
    lines = [
        f"# Experiment `{r['name']}`",
        "",
        f"- Code revision: `{r['code_revision']}`",
        f"- Matches: {', '.join(r['config']['matches'])} (leave-one-match-out)",
        f"- Attempted passes: {r['dataset']['n']} "
        f"(completion rate {r['dataset']['success_rate']:.3f}); label sources "
        f"{r['dataset']['by_label_source']}",
        f"- Receiver inference (completed passes): "
        f"{r['dataset']['receiver_inference']['accuracy']:.3f} top-1 "
        f"(n={r['dataset']['receiver_inference']['n_completed']})",
        f"- Runtime: {r['runtime_s']} s",
        "",
        "## Pooled out-of-fold metrics",
        "",
        "| model | family | log loss | Brier | AUC | ECE |",
        "|---|---|---|---|---|---|",
    ]
    for n, m in sorted(r["pooled_oof_metrics"].items(), key=lambda kv: kv[1]["log_loss"]):
        lines.append(
            f"| {n} | {r['models'][n]['family']} | {m['log_loss']:.4f} | {m['brier']:.4f} | "
            f"{m['auc']:.3f} | {m['ece']:.3f} |"
        )
    lines += ["", "## Per-fold log loss (test match)", ""]
    matches = r["config"]["matches"]
    lines.append("| model | " + " | ".join(matches) + " |")
    lines.append("|---|" + "---|" * len(matches))
    for n, fm in r["fold_metrics"].items():
        lines.append(f"| {n} | " + " | ".join(f"{fm[m]['log_loss']:.4f}" for m in matches) + " |")
    if r["comparisons"]:
        lines += [
            "",
            "## Paired comparisons (b − a; possession-clustered bootstrap, 95% CI)",
            "",
            "| a | b | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |",
            "|---|---|---|---|---|",
        ]
        for c in r["comparisons"]:

            def f(k: str, c: dict[str, Any] = c) -> str:
                lo, hi = c[k]["ci95"]
                return f"{c[k]['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]"

            lines.append(f"| {c['a']} | {c['b']} | {f('log_loss')} | {f('brier')} | {f('auc')} |")
    for n, imp in r["permutation_importance"].items():
        lines += ["", f"## Held-out permutation importance: {n} (Δ log loss, top 10)", ""]
        lines += [f"- {row['feature']}: {row['importance']:+.4f}" for row in imp[:10]]
    lines += [
        "",
        "## Caveats",
        "",
        "- Only these matches are used, all from one anonymised provider sample. CIs reflect "
        "within-match sampling of possessions, not between-match or between-competition variation.",
        "- Hyperparameters were fixed a priori (see `footymind.research.evaluate`).",
        "- Failed-pass labels depend on the documented Metrica event-mapping rules.",
    ]
    return "\n".join(lines) + "\n"
