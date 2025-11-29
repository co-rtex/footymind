# path: src/footymind/models/evaluate_model.py
"""
Evaluate a trained FootyMind model on the full processed dataset.

Usage:

    cd footymind/src
    python -m footymind.models.evaluate_model

This will:
- Load data/processed/train.csv
- Load models/footymind_model.joblib
- Compute accuracy, log_loss, baseline accuracy, confusion matrix
- Save confusion matrix and feature importance plots to plots/
"""

from __future__ import annotations

import argparse
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from footymind.config import CLASS_LABELS, TARGET_COLUMN, PLOTS_DIR
from footymind.data.data_loader import load_processed_train
from footymind.models.metrics import compute_classification_metrics
from footymind.utils.logging_utils import get_logger
from footymind.utils.paths import get_model_path

logger = get_logger(__name__)


def _prepare_features_and_labels(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prepare X, y, feature_names from processed train DataFrame.

    Mirrors the logic in train_model._prepare_features_and_labels.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != TARGET_COLUMN]

    if not feature_cols:
        raise ValueError("No numeric feature columns found in processed data.")

    X = df[feature_cols].values.astype(float)
    label_to_idx = {lab: i for i, lab in enumerate(CLASS_LABELS)}
    y = df[TARGET_COLUMN].map(label_to_idx).values.astype(int)
    return X, y, feature_cols


def _load_model_artifact():
    model_path = get_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train the model first."
        )
    artifact = joblib.load(model_path)
    return artifact


def _plot_confusion_matrix(cm: np.ndarray) -> None:
    """Plot and save confusion matrix."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(CLASS_LABELS)))
    ax.set_yticks(np.arange(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    # Annotate cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", out_path)


def _plot_feature_importances(
    feature_names: list[str],
    feature_importances: np.ndarray | None,
) -> None:
    """Plot and save feature importances if available."""
    if feature_importances is None:
        logger.warning("No feature_importances found in model artifact.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "feature_importances.png"

    # Sort features by importance
    idx_sorted = np.argsort(feature_importances)[::-1]
    top_n = min(20, len(idx_sorted))
    idx_top = idx_sorted[:top_n]

    names_top = [feature_names[i] for i in idx_top]
    imps_top = feature_importances[idx_top]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_n), imps_top)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names_top)
    ax.invert_yaxis()
    ax.set_xlabel("Relative Importance")
    ax.set_title("Top Feature Importances")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved feature importances plot to %s", out_path)


def run_evaluation() -> None:
    """Main evaluation routine."""
    logger.info("Loading processed training data...")
    df = load_processed_train()
    logger.info("Loaded %d processed training rows.", len(df))

    X, y, feature_names = _prepare_features_and_labels(df)

    logger.info("Loading trained model artifact...")
    artifact = _load_model_artifact()

    model = artifact["model"]
    scaler = artifact.get("scaler")
    feature_names_artifact = artifact.get("feature_names", feature_names)
    feature_importances = artifact.get("feature_importances")

    if scaler is not None:
        X_input = scaler.transform(X)
    else:
        X_input = X

    y_pred = model.predict(X_input)
    y_proba = model.predict_proba(X_input)

    metrics = compute_classification_metrics(
        y_true=y,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=CLASS_LABELS,
    )

    logger.info("Evaluation metrics: %s", metrics)

    # Print a concise summary to stdout as well
    print("Evaluation metrics:")
    for k in ("accuracy", "log_loss", "baseline_accuracy"):
        print(f"  {k}: {metrics[k]:.4f}")

    cm = np.array(metrics["confusion_matrix"], dtype=int)
    _plot_confusion_matrix(cm)
    _plot_feature_importances(feature_names_artifact, feature_importances)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained FootyMind model."
    )
    _ = parser.parse_args()
    run_evaluation()


if __name__ == "__main__":
    main()
