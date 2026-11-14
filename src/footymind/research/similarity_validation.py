"""Is the similar-moments comparison informative? Held-out validation (pre-registered).

``python -m footymind.research.similarity_validation <provider>`` writes
``experiments/results/similarity_validation_v1[_<provider>]/results.json``.

Protocol (fixed before running): leave-one-match-out. For every pass in the held-out match,
the ``K`` nearest passes from the *other* matches (standardised on those matches) give a
completion rate, shrunk toward the training base rate with ``PRIOR_STRENGTH`` pseudo-counts:
``p = (successes + PRIOR_STRENGTH * base) / (K + PRIOR_STRENGTH)``. It is compared with the
training base rate and with the logistic model on the same features and folds
(``Bdt_tt_logreg``), using possession-clustered paired bootstrap CIs. Success criterion for
showing neighbours in the product: kNN beats the base rate (CI excludes 0). It is not
expected to beat the logistic model. Its role is a checkable comparison, not the best
predictor.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
import polars as pl

from footymind.data.store import available_matches, git_revision
from footymind.paths import EXPERIMENTS_DIR
from footymind.research.evaluate import (
    calibration_curve,
    lomo_predictions,
    metrics,
    model_specs,
    paired_bootstrap,
)
from footymind.research.pass_dataset import DatasetConfig, build
from footymind.research.similarity import SIMILARITY_FEATURES, SimilarityIndex

K = 50
PRIOR_STRENGTH = 2.0
LR_REFERENCE = "Bdt_tt_logreg"


def knn_lomo(df: pl.DataFrame, k: int = K) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold kNN completion rate, training base rate and mean neighbour distance."""
    p = np.full(df.height, np.nan)
    base = np.full(df.height, np.nan)
    dist = np.full(df.height, np.nan)
    mids = df["match_id"].to_numpy()
    y_all = df["success"].to_numpy().astype(float)
    x_all = df.select(SIMILARITY_FEATURES).to_numpy().astype(float)
    for m in np.unique(mids):
        tr, te = mids != m, mids == m
        idx = SimilarityIndex.from_rows(df.filter(pl.Series(tr)))
        y_tr = y_all[tr]
        b = float(y_tr.mean())
        zq = np.array([idx.standardise(x) for x in x_all[te]])
        # (n_te, n_tr) distances, in chunks to bound memory
        out_p, out_d = [], []
        for s in range(0, zq.shape[0], 256):
            dd = np.linalg.norm(zq[s : s + 256, None, :] - idx.z[None, :, :], axis=2)
            nn = np.argpartition(dd, k, axis=1)[:, :k]
            succ = y_tr[nn].sum(axis=1)
            out_p.append((succ + PRIOR_STRENGTH * b) / (k + PRIOR_STRENGTH))
            out_d.append(np.take_along_axis(dd, nn, axis=1).mean(axis=1))
        p[te] = np.concatenate(out_p)
        dist[te] = np.concatenate(out_d)
        base[te] = b
    return p, base, dist


def main(provider: str = "metrica", seed: int = 0, n_boot: int = 1000) -> dict[str, Any]:
    t0 = time.time()
    mids = [m for m in available_matches() if m.startswith(provider + "-")]
    df = build(mids, DatasetConfig())
    p_knn, p_base, dist = knn_lomo(df)
    spec = model_specs()[LR_REFERENCE]
    oof_lr, _, _ = lomo_predictions(df, {LR_REFERENCE: spec}, seed)
    oof = oof_lr.with_columns(pl.Series("p_knn", p_knn), pl.Series("p_train_base_rate", p_base))
    y = oof["success"].to_numpy()
    result: dict[str, Any] = {
        "name": "similarity_validation_v1" + ("" if provider == "metrica" else f"_{provider}"),
        "provider": provider,
        "code_revision": git_revision(),
        "matches": mids,
        "k": K,
        "prior_strength": PRIOR_STRENGTH,
        "features": SIMILARITY_FEATURES,
        "n": int(df.height),
        "metrics": {
            n: metrics(y, oof[f"p_{n}"].to_numpy())
            for n in ("train_base_rate", "knn", LR_REFERENCE)
        },
        "comparisons": [
            paired_bootstrap(oof, "train_base_rate", "knn", n_boot, seed),
            paired_bootstrap(oof, LR_REFERENCE, "knn", n_boot, seed),
        ],
        "calibration_knn": calibration_curve(y, p_knn),
        "mean_neighbour_distance": float(np.nanmean(dist)),
        "runtime_s": 0.0,
    }
    result["criterion_knn_beats_base_rate"] = bool(
        result["comparisons"][0]["log_loss"]["ci95"][1] < 0
    )
    result["runtime_s"] = round(time.time() - t0, 1)
    d = EXPERIMENTS_DIR / "results" / result["name"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    r = main(sys.argv[1] if len(sys.argv) > 1 else "metrica")
    print(
        json.dumps({k: r[k] for k in ("n", "metrics", "criterion_knn_beats_base_rate")}, indent=2)
    )
    for c in r["comparisons"]:
        print(c["a"], "->", c["b"], c["log_loss"])
