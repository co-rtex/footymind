"""Train and export the serving option model (``footymind train-option-model``).

1. Build the attempted-pass dataset for the given matches.
2. Evaluate the option-model specification leave-one-match-out. These are the honest
   held-out metrics, and they are stored in the model metadata.
3. Fit on all matches and export plain-JSON coefficients to
   ``src/footymind/spatial/option_model_v1.json``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from footymind.data.store import git_revision
from footymind.research.evaluate import lomo_predictions, metrics, model_specs
from footymind.research.pass_dataset import DatasetConfig, build
from footymind.spatial.option_model import MODEL_RESOURCE, OPTION_FEATURES, OptionModel

OUT = Path(__file__).resolve().parents[1] / "spatial" / MODEL_RESOURCE


def train(match_ids: list[str], out: Path = OUT, seed: int = 0) -> OptionModel:
    df = build(match_ids, DatasetConfig())
    spec = model_specs()["option_logreg"]
    oof, folds, _ = lomo_predictions(df, {"option_logreg": spec}, seed)
    y = oof["success"].to_numpy()
    pooled = metrics(y, oof["p_option_logreg"].to_numpy())
    x = df.select(OPTION_FEATURES).to_numpy().astype(float)
    assert spec.make is not None
    pipe = spec.make(seed)
    pipe.fit(x, y)
    model = OptionModel.from_pipeline(
        pipe,
        OPTION_FEATURES,
        {
            "name": "option_model_v1",
            "trained_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "code_revision": git_revision(),
            "training_matches": match_ids,
            "n_train": int(df.height),
            "train_success_rate": float(y.mean()),
            "lomo_pooled": pooled,
            "lomo_folds": folds["option_logreg"],
            "status": (
                "logistic regression calibrated on attempted passes (leave-one-match-out "
                "evaluated); applied to unchosen/hypothetical options it extrapolates "
                "under selection bias"
            ),
        },
    )
    # The exported model must reproduce the sklearn pipeline exactly.
    rows = [dict(zip(OPTION_FEATURES, r, strict=True)) for r in x[:200]]
    np.testing.assert_allclose(model.predict(rows), pipe.predict_proba(x[:200])[:, 1], atol=1e-9)
    out.write_text(model.to_json() + "\n")
    return model
