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
EXPERIMENT = "pass_outcome_v3"
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
    r = _load(EXPERIMENT)
    curves = [
        ("A_logreg", "Event-only (A)"),
        ("B_logreg", "Spatial (B)"),
        ("physics_raw", "Physics, uncalibrated"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.plot(
        [0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=(0, (3, 3)), label="Perfect calibration"
    )
    for (key, label), color in zip(curves, SERIES, strict=True):
        pts = [p for p in r["calibration"][key] if p["n"] >= 15]
        xs = [p["mean_pred"] for p in pts]
        ys = [p["observed"] for p in pts]
        ece = r["pooled_oof_metrics"][key]["ece"]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=f"{label} · ECE {ece:.3f}",
        )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Predicted completion probability (bin mean)", color=INK, fontsize=9)
    ax.set_ylabel("Observed completion rate", color=INK, fontsize=9)
    ax.set_title(
        "Calibration on held-out matches (LOMO, 3 matches, 3,593 passes)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def ablation(path: Path) -> None:
    r = _load(EXPERIMENT)
    names = {
        ("Adt_logreg", "Bdt_logreg"): "Decision-time: event → + spatial (LR)",
        ("Adt_gbm", "Bdt_gbm"): "Decision-time: event → + spatial (GBM)",
        ("A_logreg", "B_logreg"): "Action-conditioned: event → + spatial (LR)",
        ("A_gbm", "B_gbm"): "Action-conditioned: event → + spatial (GBM)",
        ("A_logreg", "abl_A+target_geometry_logreg"): "Action-cond.: + teammate geometry",
        ("A_logreg", "abl_A+defender_only_logreg"): "Action-cond.: + defender-only",
        ("B_gbm", "C_gbm"): "Spatial → + interaction (GBM)",
        ("option_logreg", "option_v2_logreg"): "Option model v1 → v2 (lead pass)",
    }
    rows = [
        (names[(c["a"], c["b"])], c["log_loss"])
        for c in r["comparisons"]
        if (c["a"], c["b"]) in names
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    ax.axvline(0, color=INK_2, linewidth=1)
    for i, (_label, ll) in enumerate(rows):
        lo, hi = ll["ci95"]
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=SERIES[0], linewidth=2, solid_capstyle="round")
        ax.plot(
            [ll["delta"]],
            [y],
            "o",
            color=SERIES[0],
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        ax.text(max(hi, 0) + 0.003, y, f"{ll['delta']:+.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8, color=INK)
    ax.set_xlabel("Δ held-out log loss (95% CI; left of 0 = better)", color=INK, fontsize=9)
    ax.set_title(
        "Which information improves pass-outcome prediction?", color=INK, fontsize=10, loc="left"
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def folds(path: Path) -> None:
    r = _load(EXPERIMENT)
    matches = r["config"]["matches"]
    models = [
        ("Adt_logreg", "Decision-time event"),
        ("Bdt_logreg", "Decision-time spatial"),
        ("B_logreg", "Action-conditioned spatial"),
    ]
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
    ax.set_ylim(0, max(r["fold_metrics"][models[0][0]][m]["log_loss"] for m in matches) * 1.2)
    ax.set_title(
        "Leave-one-match-out log loss by held-out match", color=INK, fontsize=10, loc="left"
    )
    ax.legend(frameon=False, fontsize=8, ncols=3, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def replication(path: Path) -> None:
    """Same comparisons on two providers, side by side (never pooled)."""
    runs = [
        (EXPERIMENT, "Metrica (3 matches)"),
        ("pass_outcome_skillcorner_v2", "SkillCorner (20 matches)"),
    ]
    comps = [
        (("Adt_logreg", "Bdt_logreg"), "Decision-time: + spatial"),
        (("A_logreg", "B_logreg"), "Action-conditioned: + spatial"),
        (("A_logreg", "abl_A+defender_only_logreg"), "Action-cond.: + defender-only"),
        (("B_gbm", "C_gbm"), "Spatial → + interaction (GBM)"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    ax.axvline(0, color=INK_2, linewidth=1)
    for k, ((name, label), color) in enumerate(zip(runs, SERIES, strict=False)):
        r = _load(name)
        by = {(c["a"], c["b"]): c["log_loss"] for c in r["comparisons"]}
        for i, (pair, _) in enumerate(comps):
            ll = by[pair]
            y = len(comps) - 1 - i + (0.18 if k == 0 else -0.18)
            lo, hi = ll["ci95"]
            ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
            ax.plot(
                [ll["delta"]],
                [y],
                "o",
                color=color,
                markersize=5.5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=label if i == 0 else None,
            )
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels([c[1] for c in comps][::-1], fontsize=8, color=INK)
    ax.set_xlabel("Δ held-out log loss (95% CI; left of 0 = better)", color=INK, fontsize=9)
    ax.set_title(
        "Cross-provider replication (separate LOMO evaluations, never pooled)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def rq2(path: Path) -> None:
    """RQ2 temporal context: release frame vs 0.2 s guard, per provider (never pooled).

    Colour = provider; filled marker = guarded run (ADR-0007), hollow = release frame.
    """
    runs = [
        ("temporal_v1", "Metrica, release frame", SERIES[0], False),
        ("temporal_guard_v1_02s", "Metrica, 0.2 s guard", SERIES[0], True),
        ("temporal_skillcorner_v1", "SkillCorner, release frame", SERIES[1], False),
        (
            "temporal_guard_skillcorner_v1_02s",
            "SkillCorner, 0.2 s guard (primary)",
            SERIES[1],
            True,
        ),
    ]
    comps = [
        (("Adt_logreg", "Bdt_logreg"), "+ spatial context (C20/C28)"),
        (("Adt_tob_logreg", "Bdt_tob_logreg"), "+ spatial, vs time-on-ball baseline (H-T3)"),
        (("Adt_logreg", "Adt_tob_logreg"), "+ time-on-ball (H-T1)"),
        (("Bdt_tob_logreg", "Bdt_tt_logreg"), "+ 1 s movement history (H-T2)"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    ax.axvline(0, color=INK_2, linewidth=1)
    offsets = [0.27, 0.09, -0.09, -0.27]
    for k, (name, label, color, guarded) in enumerate(runs):
        try:
            r = _load(name)
        except FileNotFoundError:
            continue
        by = {(c["a"], c["b"]): c["log_loss"] for c in r["comparisons"]}
        first = True
        for i, (pair, _) in enumerate(comps):
            if pair not in by:
                continue
            ll = by[pair]
            y = len(comps) - 1 - i + offsets[k]
            lo, hi = ll["ci95"]
            ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
            ax.plot(
                [ll["delta"]],
                [y],
                "o",
                color=color,
                markersize=6,
                markerfacecolor=color if guarded else SURFACE,
                markeredgecolor=color,
                markeredgewidth=1.6,
                label=label if first else None,
            )
            first = False
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels([c[1] for c in comps][::-1], fontsize=8, color=INK)
    ax.set_xlabel("Δ held-out log loss (95% CI; left of 0 = better)", color=INK, fontsize=9)
    ax.set_title(
        "RQ2: temporal context, release frame vs guard frame (LOMO, never pooled)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reliability(OUT / "calibration_v3.png")
    ablation(OUT / "ablation_v3.png")
    folds(OUT / "folds_v3.png")
    replication(OUT / "replication_v1.png")
    rq2(OUT / "rq2_temporal_v1.png")
    print(sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
