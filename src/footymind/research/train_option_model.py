"""Train and export the serving option model (``footymind train-option-model``).

1. Build the attempted-pass dataset for the given matches.
2. Evaluate the option-model specification leave-one-match-out. These are the honest
   held-out metrics, and they are stored in the model metadata.
3. Fit on all matches and export plain-JSON coefficients to
   ``src/footymind/spatial/option_model_v2.json``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from footymind.data.store import git_revision
from footymind.research.evaluate import lomo_predictions, metrics, model_specs
from footymind.research.pass_dataset import DatasetConfig, build
from footymind.spatial.option_model import MODEL_RESOURCE, SERVING_FEATURES, OptionModel

OUT = Path(__file__).resolve().parents[1] / "spatial" / MODEL_RESOURCE


def train(match_ids: list[str], out: Path = OUT, seed: int = 0, n_boot: int = 32) -> OptionModel:
    df = build(match_ids, DatasetConfig(rules="v2"))
    spec = model_specs()["option_v2_logreg"]
    oof, folds, _ = lomo_predictions(df, {"option_v2_logreg": spec}, seed)
    y = oof["success"].to_numpy()
    pooled = metrics(y, oof["p_option_v2_logreg"].to_numpy())
    x = df.select(SERVING_FEATURES).to_numpy().astype(float)
    assert spec.make is not None
    pipe = spec.make(seed)
    pipe.fit(x, y)
    # Possession-clustered bootstrap of the logistic coefficients, with the preprocessing
    # (imputer, scaler) held fixed. Passes in one possession are correlated, so
    # resampling is by possession.
    z = pipe[1].transform(pipe[0].transform(x))
    groups = (df["match_id"] + "|" + df["possession_id"].fill_null("none")).to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(uniq.size)]
    rng = np.random.default_rng(seed)
    boot_c, boot_i = [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, uniq.size, uniq.size)])
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(z[idx], y[idx])
        boot_c.append(lr.coef_[0])
        boot_i.append(lr.intercept_[0])
    model = OptionModel.from_pipeline(
        pipe,
        SERVING_FEATURES,
        {
            "name": "option_model_v2",
            "dataset_rules": "v2",
            "trained_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "code_revision": git_revision(),
            "training_matches": match_ids,
            "n_train": int(df.height),
            "train_success_rate": float(y.mean()),
            "lomo_pooled": pooled,
            "lomo_folds": folds["option_v2_logreg"],
            "n_bootstrap": n_boot,
            "status": (
                "logistic regression calibrated on attempted passes (leave-one-match-out "
                "evaluated); applied to unchosen/hypothetical options it extrapolates "
                "under selection bias"
            ),
        },
    )
    model = OptionModel(
        **{**model.__dict__, "boot_coef": np.array(boot_c), "boot_intercept": np.array(boot_i)}
    )
    # The exported model must reproduce the sklearn pipeline exactly.
    rows = [dict(zip(SERVING_FEATURES, r, strict=True)) for r in x[:200]]
    np.testing.assert_allclose(model.predict(rows), pipe.predict_proba(x[:200])[:, 1], atol=1e-9)
    out.write_text(model.to_json() + "\n")
    return model
