"""
Explainability utilities for FootyMind.

Provides:
- A simple, fast per-instance explanation based on feature importances and
  deviation from global feature means (used by API/UI).
- An optional CLI that uses SHAP to generate a summary plot for deeper
  offline analysis.

Usage (for SHAP summary):

    python -m footymind.models.explain_model --shap-summary

Requires SHAP to be installed (already in requirements.txt).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from footymind.config import CLASS_LABELS, PLOTS_DIR
from footymind.data.data_loader import load_processed_train
from footymind.utils.logging_utils import get_logger
from footymind.utils.paths import get_model_path

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Simple local explanation (used by API/UI)
# ---------------------------------------------------------------------------


def simple_local_explanation(
    feature_vector: np.ndarray,
    feature_names: Sequence[str],
    feature_means: np.ndarray | None,
    feature_importances: np.ndarray | None,
    top_n: int = 8,
) -> List[Dict[str, Any]]:
    """
    Compute a simple local explanation for a single instance.

    Uses:
    - Deviation from global feature means.
    - Global feature importances (normalized) if available.

    Contribution is approximated as:
        contribution_i = feature_importance_i * (value_i - mean_i)

    Parameters
    ----------
    feature_vector : np.ndarray
        Feature values for a single instance (shape: [n_features]).
    feature_names : Sequence[str]
        Names of each feature in order.
    feature_means : np.ndarray | None
        Global means for each feature (same order). If None, uses zeros.
    feature_importances : np.ndarray | None
        Global feature importance scores (nonnegative). If None, uses ones.
    top_n : int
        Number of top-contributing features to return.

    Returns
    -------
    List[Dict[str, Any]]
        List of feature contribution dicts sorted by absolute contribution
        (descending). Each dict contains:
        - feature_name
        - value
        - mean
        - importance
        - contribution
        - direction ("positive" or "negative")
    """
    x = np.asarray(feature_vector, dtype=float)
    n = x.shape[0]

    if feature_means is None:
        means = np.zeros_like(x)
    else:
        means = np.asarray(feature_means, dtype=float)
        if means.shape[0] != n:
            raise ValueError(
                "feature_means length does not match feature_vector.")

    if feature_importances is None:
        imps = np.ones_like(x)
    else:
        imps = np.asarray(feature_importances, dtype=float)
        if imps.shape[0] != n:
            raise ValueError(
                "feature_importances length does not match feature_vector.")

    deltas = x - means
    contributions = imps * deltas
    abs_contrib = np.abs(contributions)

    # Sort by absolute contribution descending
    idx_sorted = np.argsort(abs_contrib)[::-1]
    top_n = min(top_n, n)
    idx_top = idx_sorted[:top_n]

    explanations: List[Dict[str, Any]] = []
    for idx in idx_top:
        contrib = float(contributions[idx])
        direction = "positive" if contrib >= 0 else "negative"
        explanations.append(
            {
                "feature_name": str(feature_names[idx]),
                "value": float(x[idx]),
                "mean": float(means[idx]),
                "importance": float(imps[idx]),
                "contribution": contrib,
                "direction": direction,
            }
        )

    return explanations


# ---------------------------------------------------------------------------
# Optional SHAP-based explanation (offline CLI)
# ---------------------------------------------------------------------------


def _load_model_artifact() -> Dict[str, Any]:
    path = get_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Train the model first."
        )
    return joblib.load(path)


def _prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feature_cols = [c for c in df.columns if c != "outcome"]
    X = df[feature_cols].values.astype(float)
    return X, feature_cols


def generate_shap_summary(
    max_samples: int = 200,
    out_path: Path | None = None,
) -> None:
    """
    Generate a SHAP summary plot on a subset of the training data.

    Parameters
    ----------
    max_samples : int
        Maximum number of samples from the dataset to use for SHAP.
    out_path : pathlib.Path | None
        Where to save the plot. If None, uses plots/shap_summary.png.
    """
    try:
        import shap  # type: ignore[import-not-found]
    except ImportError:
        logger.error(
            "SHAP is not installed. Install it or check requirements.txt."
        )
        return

    logger.info("Loading processed training data...")
    df = load_processed_train()
    X, feature_names = _prepare_features(df)

    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], size=max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    artifact = _load_model_artifact()
    model = artifact["model"]
    scaler = artifact.get("scaler")

    if scaler is not None:
        X_sample_input = scaler.transform(X_sample)
    else:
        X_sample_input = X_sample

    logger.info("Computing SHAP values on %d samples...",
                X_sample_input.shape[0])

    # For tree-based models, TreeExplainer is efficient; otherwise fallback.
    if artifact["model_type"] == "gradient_boosting":
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_sample_input)

    shap_values = explainer(X_sample_input)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = PLOTS_DIR / "shap_summary.png"

    shap.plots.beeswarm(shap_values, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved SHAP summary plot to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain trained FootyMind model."
    )
    parser.add_argument(
        "--shap-summary",
        action="store_true",
        help="Generate SHAP summary plot on a subset of training data.",
    )
    args = parser.parse_args()

    if args.shap_summary:
        generate_shap_summary()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
