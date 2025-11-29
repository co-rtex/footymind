"""
Train FootyMind models on processed match features.

Usage (from project root):

    cd src
    python -m footymind.models.train_model

This will:
- Load data/processed/train.csv
- Split into train/validation sets
- Train multiple candidate models:
    * Logistic Regression
    * Gradient Boosting
    * Random Forest
- Select the best model (by log loss, with accuracy as fallback)
- Save the best model artifact to models/footymind_model.joblib
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from footymind.config import CLASS_LABELS, TARGET_COLUMN
from footymind.data.data_loader import load_processed_train
from footymind.models.metrics import compute_classification_metrics
from footymind.utils.logging_utils import get_logger
from footymind.utils.paths import get_model_path

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    """
    Configuration for the training process.

    Attributes
    ----------
    test_size : float
        Fraction of data reserved for validation.
    random_state : int
        Random seed for reproducibility.
    use_scaler : bool
        Whether to apply StandardScaler to features (recommended for LogReg).
    """

    test_size: float = 0.25
    random_state: int = 42
    use_scaler: bool = True


def _prepare_features_and_labels(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare X, y, feature_names from processed train DataFrame.

    This mirrors the logic used in evaluate_model._prepare_features_and_labels.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != TARGET_COLUMN]

    if not feature_cols:
        raise ValueError("No numeric feature columns found in processed data.")

    X = df[feature_cols].values.astype(float)

    label_to_idx = {lab: i for i, lab in enumerate(CLASS_LABELS)}
    y = df[TARGET_COLUMN].map(label_to_idx).values.astype(int)

    return X, y, feature_cols


def _get_feature_importances_from_model(
    model: Any, feature_names: List[str]
) -> np.ndarray:
    """
    Derive a 1D feature importance array from a fitted model.

    - If the model has `feature_importances_` (tree-based models), use that.
    - If the model is LogisticRegression, use mean absolute coefficients.
    - Otherwise, fall back to uniform importances (all ones).
    """
    # Tree-based models
    if hasattr(model, "feature_importances_"):
        imps = np.asarray(model.feature_importances_, dtype=float)
        if imps.shape[0] == len(feature_names):
            return imps

    # Logistic regression: use mean abs coefficients across classes
    if isinstance(model, LogisticRegression):
        coefs = np.asarray(model.coef_, dtype=float)
        if coefs.ndim == 1:
            coefs = coefs.reshape(1, -1)
        if coefs.shape[1] == len(feature_names):
            imps = np.mean(np.abs(coefs), axis=0)
            return imps

    # Fallback: uniform
    logger.warning(
        "Could not infer feature_importances for model %s; "
        "falling back to uniform importances.",
        type(model).__name__,
    )
    return np.ones(len(feature_names), dtype=float)


def run_training(cfg: TrainConfig) -> None:
    """
    Run the end-to-end training process and save the best model artifact.
    """
    logger.info("Loading processed training data...")
    df = load_processed_train()
    logger.info("Loaded %d processed training rows.", len(df))
    logger.info("Data shape: %s", (df.shape[0], df.shape[1]))

    if len(df) < 3:
        raise ValueError(
            "Not enough rows in processed training data to perform "
            "a train/validation split. Please use more data."
        )

    X, y, feature_names = _prepare_features_and_labels(df)

    # Check class distribution
    class_counts = np.bincount(y, minlength=len(CLASS_LABELS))
    logger.info("Class counts: %s", class_counts.tolist())

    # For tiny datasets, stratified split can fail if a class has < 2 samples
    if np.any(class_counts < 2):
        logger.warning(
            "Not using stratified split due to small sample per class. "
            "Class counts: %s",
            class_counts.tolist(),
        )
        stratify = None
    else:
        stratify = y

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify,
    )

    logger.info(
        "Train/val split: train=%d, val=%d", X_train.shape[0], X_val.shape[0]
    )

    scaler = None
    if cfg.use_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    # Candidate models
    candidates: List[Tuple[str, Any]] = []

    # Logistic Regression
    logreg = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        random_state=cfg.random_state,
    )
    candidates.append(("logistic_regression", logreg))

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        random_state=cfg.random_state,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
    )
    candidates.append(("gradient_boosting", gb))

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    candidates.append(("random_forest", rf))

    results: List[Dict[str, Any]] = []

    for name, model in candidates:
        logger.info("Training %s...", name)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        metrics = compute_classification_metrics(
            y_true=y_val,
            y_pred=y_pred,
            y_proba=y_proba,
            labels=CLASS_LABELS,
        )
        logger.info("%s metrics: %s", name, metrics)

        results.append(
            {
                "name": name,
                "model": model,
                "metrics": metrics,
            }
        )

    # Select best model: prefer lowest log_loss if defined, else highest accuracy
    def _score_for_selection(res: Dict[str, Any]) -> Tuple[float, float]:
        m = res["metrics"]
        ll = m.get("log_loss", float("nan"))
        acc = m.get("accuracy", float("nan"))

        if np.isnan(ll):
            # log_loss not available / meaningful -> treat it as very bad
            return (float("inf"), -acc if not np.isnan(acc) else float("inf"))
        return (ll, -acc if not np.isnan(acc) else 0.0)

    results_sorted = sorted(results, key=_score_for_selection)
    best = results_sorted[0]

    best_name = best["name"]
    best_model = best["model"]
    best_metrics = best["metrics"]

    logger.info("Selected %s as best model.", best_name)
    logger.info("Best model metrics: %s", best_metrics)

    # Compute global feature means (on full dataset, not just train)
    feature_means = X.mean(axis=0)

    # Compute feature importances for the best model
    feature_importances = _get_feature_importances_from_model(
        best_model, feature_names
    )

    # Build artifact
    artifact: Dict[str, Any] = {
        "model": best_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "class_labels": CLASS_LABELS,
        "feature_means": feature_means,
        "feature_importances": feature_importances,
        "training_metrics": {
            best_name: best_metrics,
        },
    }

    model_path = get_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    logger.info(
        "Saved best model (%s) to %s", best_name, model_path
    )


def main() -> None:
    cfg = TrainConfig()
    run_training(cfg)


if __name__ == "__main__":
    main()
