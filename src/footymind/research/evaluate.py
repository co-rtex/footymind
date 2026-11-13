"""Model families, leave-one-match-out evaluation and uncertainty for pass outcomes.

Methodological commitments (fixed before seeing results, see docs/RESEARCH.md):

* Splits are leave-one-match-out: a test match never contributes training rows.
* Hyperparameters are fixed a priori. No tuning is done on test folds, and
  there is no nested tuning because the data are too small to tune credibly.
* Uncertainty: paired bootstrap over *possessions* (passes within a
  possession are correlated) on pooled out-of-fold predictions. With only
  2-3 matches, between-match variance cannot be estimated, so CIs reflect
  within-match sampling only. They are not population-level guarantees.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from footymind.research.pass_dataset import (
    EVENT_FEATURES,
    GROUP_PRESSURE,
    GROUP_RAY_DEFENSE,
    GROUP_TARGET_DEFENSE,
    GROUP_TARGET_GEOMETRY,
    INTERACTION_FEATURES,
    SPATIAL_FEATURES,
)

EPS = 1e-4


def _logreg() -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000),
    )


def _gbm(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=seed,
    )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str  # "baseline" | "A" | "B" | "C" | "physics"
    features: list[str]
    make: Callable[[int], Any] | None  # None -> non-fitted rule
    description: str


def model_specs() -> dict[str, ModelSpec]:
    a, b = EVENT_FEATURES, EVENT_FEATURES + SPATIAL_FEATURES
    c = b + INTERACTION_FEATURES
    specs = [
        ModelSpec("base_rate", "baseline", [], None, "training-set completion rate"),
        ModelSpec("A_logreg", "A", a, lambda s: _logreg(), "event features, logistic regression"),
        ModelSpec("A_gbm", "A", a, _gbm, "event features, gradient boosting"),
        ModelSpec(
            "physics_raw",
            "physics",
            ["cand_p_success_physics"],
            None,
            "uncalibrated PPCF x lane-clear estimate for the inferred receiver (no fitting)",
        ),
        ModelSpec(
            "physics_platt",
            "physics",
            ["logit_physics"],
            lambda s: _logreg(),
            "physics estimate recalibrated with a 1-feature logistic regression",
        ),
        ModelSpec("B_logreg", "B", b, lambda s: _logreg(), "event + spatial, logistic regression"),
        ModelSpec("B_gbm", "B", b, _gbm, "event + spatial, gradient boosting"),
        ModelSpec("C_logreg", "C", c, lambda s: _logreg(), "event + spatial + interaction, LR"),
        ModelSpec("C_gbm", "C", c, _gbm, "event + spatial + interaction, gradient boosting"),
    ]
    a_ = EVENT_FEATURES
    ablations = {
        "A+pressure": a_ + GROUP_PRESSURE,
        "A+ray_defense": a_ + GROUP_RAY_DEFENSE,
        "A+defender_only": a_ + GROUP_PRESSURE + GROUP_RAY_DEFENSE,
        "A+target_geometry": a_ + GROUP_TARGET_GEOMETRY,
        "A+target_geometry+target_defense": a_ + GROUP_TARGET_GEOMETRY + GROUP_TARGET_DEFENSE,
        "B_minus_target_geometry": [f for f in b if f not in GROUP_TARGET_GEOMETRY],
    }
    for n, feats in ablations.items():
        specs.append(ModelSpec(f"abl_{n}_logreg", "ablation", feats, lambda s: _logreg(), n))
    return {s.name: s for s in specs}


def _prepare(df: pl.DataFrame) -> pl.DataFrame:
    p = df["cand_p_success_physics"].clip(EPS, 1 - EPS)
    return df.with_columns((p / (1 - p)).log().alias("logit_physics"))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for k in range(n_bins):
        m = bins == k
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, EPS, 1 - EPS)
    return {
        "n": int(y.size),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "ece": ece(y, p),
        "mean_pred": float(p.mean()),
        "base_rate": float(y.mean()),
    }


def calibration_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    out = []
    for k in range(n_bins):
        m = bins == k
        if m.any():
            out.append(
                {
                    "bin": k,
                    "n": int(m.sum()),
                    "mean_pred": float(p[m].mean()),
                    "observed": float(y[m].mean()),
                }
            )
    return out


def lomo_predictions(
    df: pl.DataFrame, specs: dict[str, ModelSpec], seed: int = 0
) -> tuple[pl.DataFrame, dict[str, dict[str, dict[str, float]]], dict[str, Any]]:
    """Out-of-fold predictions for every model under leave-one-match-out."""
    df = _prepare(df)
    matches = sorted(df["match_id"].unique().to_list())
    preds = {name: np.full(df.height, np.nan) for name in specs}
    fold_metrics: dict[str, dict[str, dict[str, float]]] = {name: {} for name in specs}
    fitted: dict[str, Any] = {}
    y_all = df["success"].to_numpy()
    for test_m in matches:
        te = (df["match_id"] == test_m).to_numpy()
        tr = ~te
        for name, spec in specs.items():
            if spec.name == "base_rate":
                p = np.full(te.sum(), y_all[tr].mean())
            elif spec.make is None:
                p = df.filter(pl.Series(te))[spec.features[0]].to_numpy().astype(float)
                p = np.clip(np.nan_to_num(p, nan=y_all[tr].mean()), EPS, 1 - EPS)
            else:
                x_tr = df.filter(pl.Series(tr)).select(spec.features).to_numpy().astype(float)
                x_te = df.filter(pl.Series(te)).select(spec.features).to_numpy().astype(float)
                model = spec.make(seed)
                model.fit(x_tr, y_all[tr])
                p = model.predict_proba(x_te)[:, 1]
                fitted[(name, test_m)] = (model, x_te, y_all[te])  # type: ignore[index]
            preds[name][te] = p
            fold_metrics[name][test_m] = metrics(y_all[te], p)
    out = df.select("match_id", "possession_id", "event_id", "success").with_columns(
        [pl.Series(f"p_{n}", v) for n, v in preds.items()]
    )
    return out, fold_metrics, fitted


def paired_bootstrap(
    oof: pl.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """CI for metric(b) - metric(a), resampling possessions within each match."""
    rng = np.random.default_rng(seed)
    y = oof["success"].to_numpy()
    pa = np.clip(oof[f"p_{a}"].to_numpy(), EPS, 1 - EPS)
    pb = np.clip(oof[f"p_{b}"].to_numpy(), EPS, 1 - EPS)
    groups = (oof["match_id"] + "|" + oof["possession_id"].fill_null("none")).to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(uniq.size)]
    match_of_group = np.array([u.split("|")[0] for u in uniq])
    strata = [np.flatnonzero(match_of_group == m) for m in np.unique(match_of_group)]

    def ll(yy: np.ndarray, pp: np.ndarray) -> float:
        return float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)))

    def br(yy: np.ndarray, pp: np.ndarray) -> float:
        return float(np.mean((yy - pp) ** 2))

    point = {"log_loss": ll(y, pb) - ll(y, pa), "brier": br(y, pb) - br(y, pa)}
    try:
        point["auc"] = float(roc_auc_score(y, pb) - roc_auc_score(y, pa))
    except ValueError:
        point["auc"] = float("nan")
    boots: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        idx = np.concatenate(
            [np.concatenate([members[g] for g in rng.choice(s, size=s.size)]) for s in strata]
        )
        yy = y[idx]
        boots["log_loss"].append(ll(yy, pb[idx]) - ll(yy, pa[idx]))
        boots["brier"].append(br(yy, pb[idx]) - br(yy, pa[idx]))
        if len(np.unique(yy)) == 2:
            boots["auc"].append(roc_auc_score(yy, pb[idx]) - roc_auc_score(yy, pa[idx]))
    res: dict[str, Any] = {"a": a, "b": b, "n_boot": n_boot, "n_groups": int(uniq.size)}
    for k, v in boots.items():
        arr = np.array(v)
        res[k] = {
            "delta": point[k],
            "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
            "share_improved": float(np.mean(arr < 0) if k != "auc" else np.mean(arr > 0)),
        }
    return res


def permutation_importances(
    fitted: dict[Any, Any], spec: ModelSpec, seed: int = 0, n_repeats: int = 10
) -> list[dict[str, float]]:
    """Held-out permutation importance (increase in log loss), averaged across folds."""
    acc: dict[str, list[float]] = {f: [] for f in spec.features}
    for (name, _m), (model, x_te, y_te) in fitted.items():
        if name != spec.name:
            continue
        r = permutation_importance(
            model, x_te, y_te, scoring="neg_log_loss", n_repeats=n_repeats, random_state=seed
        )
        for f, v in zip(spec.features, r.importances_mean, strict=True):
            acc[f].append(float(v))
    rows: list[dict[str, float]] = [
        {"feature": f, "importance": float(np.mean(v))}  # type: ignore[dict-item]
        for f, v in acc.items()
        if v
    ]
    return sorted(rows, key=lambda r: -r["importance"])
