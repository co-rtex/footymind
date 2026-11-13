"""Static research figures generated from committed ``results.json`` files.

``uv run --group research python -m footymind.research.figures`` writes PNGs into
``docs/figures/``. Every number plotted is read from a committed results file, and
nothing is recomputed here, so figures cannot drift from the claims ledger.

Palette: the validated reference categorical slots 1-3 (blue, orange, aqua), which
pass all-pairs CVD checks on a light surface. The legend names every series (with its
ECE). Because aqua needs contrast relief, the same numbers also appear as tables in
docs/RESEARCH.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from footymind.paths import EXPERIMENTS_DIR, REPO_ROOT

OUT = REPO_ROOT / "docs" / "figures"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _load(name: str) -> dict[str, Any]:
    return json.loads((EXPERIMENTS_DIR / "results" / name / "results.json").read_text())


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def reliability(path: Path) -> None:
    r = _load("pass_outcome_v2_3match")
    curves = [
        ("A_logreg", "Event-only (A)"),
        ("B_logreg", "Spatial (B)"),
        ("physics_raw", "Physics, uncalibrated"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=(0, (3, 3)),
            label="Perfect calibration")
    for (key, label), color in zip(curves, SERIES, strict=True):
        pts = [p for p in r["calibration"][key] if p["n"] >= 15]
        xs = [p["mean_pred"] for p in pts]
        ys = [p["observed"] for p in pts]
        ece = r["pooled_oof_metrics"][key]["ece"]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{label} · ECE {ece:.3f}")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Predicted completion probability (bin mean)", color=INK, fontsize=9)
    ax.set_ylabel("Observed completion rate", color=INK, fontsize=9)
    ax.set_title(
        "Calibration on held-out matches (LOMO, 3 matches, 3,633 passes)",
        color=INK, fontsize=10, loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def ablation(path: Path) -> None:
    r = _load("pass_outcome_v2_3match")
    names = {
        ("A_logreg", "B_logreg"): "Event → + all spatial (LR)",
        ("A_gbm", "B_gbm"): "Event → + all spatial (GBM)",
        ("A_logreg", "abl_A+target_geometry_logreg"): "Event → + teammate geometry",
        ("A_logreg", "abl_A+defender_only_logreg"): "Event → + defender-only",
        ("B_gbm", "C_gbm"): "Spatial → + interaction (GBM)",
        ("A_logreg", "physics_platt"): "Event → physics (Platt) instead",
    }
    rows = [(names[(c["a"], c["b"])], c["log_loss"]) for c in r["comparisons"]
            if (c["a"], c["b"]) in names]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    ax.axvline(0, color=INK_2, linewidth=1)
    for i, (_label, ll) in enumerate(rows):
        lo, hi = ll["ci95"]
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=SERIES[0], linewidth=2, solid_capstyle="round")
        ax.plot([ll["delta"]], [y], "o", color=SERIES[0], markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(max(hi, 0) + 0.003, y, f"{ll['delta']:+.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8, color=INK)
    ax.set_xlabel("Δ held-out log loss (95% CI; left of 0 = better)", color=INK, fontsize=9)
    ax.set_title("Which information improves pass-outcome prediction?",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def folds(path: Path) -> None:
    r = _load("pass_outcome_v2_3match")
    matches = r["config"]["matches"]
    models = [("A_logreg", "Event-only (A)"), ("B_logreg", "Spatial (B)"),
              ("option_logreg", "Option model")]
    models = [m for m in models if m[0] in r["fold_metrics"]]
    if len(models) < 3:  # the option model is not part of v2; fall back to B_gbm
        models.append(("B_gbm", "Spatial (B, GBM)"))
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.grid(False, axis="x")
    width = 0.26
    for k, ((key, label), color) in enumerate(zip(models, SERIES, strict=False)):
        xs = [i + (k - 1) * (width + 0.02) for i in range(len(matches))]
        vals = [r["fold_metrics"][key][m]["log_loss"] for m in matches]
        ax.bar(xs, vals, width=width, color=color, label=label)
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(matches)))
    ax.set_xticklabels(["Game 1", "Game 2", "Game 3 (EPTS, transfer)"], fontsize=8, color=INK)
    ax.set_ylabel("Held-out log loss (lower = better)", color=INK, fontsize=9)
    ax.set_ylim(0, max(r["fold_metrics"]["A_logreg"][m]["log_loss"] for m in matches) * 1.18)
    ax.set_title("Leave-one-match-out: every held-out match improves",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, ncols=3, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reliability(OUT / "calibration_v2.png")
    ablation(OUT / "ablation_v2.png")
    folds(OUT / "folds_v2.png")
    print(sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
