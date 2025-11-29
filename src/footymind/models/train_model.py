# path: src/footymind/models/train_model.py
"""
Train FootyMind models on processed data.

Usage:

    python -m footymind.models.train_model

This will:
- Load data/processed/train.csv.
- Train a baseline Logistic Regression and a Gradient Boosting model.
- Compare performance on a validation split.
- Save the best-performing model to models/footymind_model.joblib.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from footymind.config import CLASS_LABELS, RANDOM_STATE, TARGET_COLUMN
from footymind.data.data_loader import load_processed_train
from footymind.models.metrics import compute_classification_metrics
from footymind.utils.logging_utils import get_logger
from footymind.utils.paths import get_model_path

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    """Configuration for model training."""

    test_size: float = 0.25
    random_state: int = RANDOM_STATE


def _prepare_features_and_labels(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Split processed DataFrame into X (features) and y (class indices).

    Only numeric columns (excluding the label) are used as features.

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Label indices (0..n_classes-1).
    feature_names : list[str]
        Names of features (columns corresponding to X).
    """
    # Keep only numeric columns that are not the target
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != TARGET_COLUMN]

    if not feature_cols:
        raise ValueError("No numeric feature columns found in processed data.")

    X = df[feature_cols].values.astype(float)
    label_to_idx = {lab: i for i, lab in enumerate(CLASS_LABELS)}
    y = df[TARGET_COLUMN].map(label_to_idx).values.astype(int)
    return X, y, feature_cols


def _train_log_reg(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[LogisticRegression, StandardScaler]:
    """Train a baseline multinomial logistic regression with scaling."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(
        multi_class="multinomial",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_scaled, y_train)
    return clf, scaler


def _train_gb(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> GradientBoostingClassifier:
    """Train a Gradient Boosting classifier."""
    gb = GradientBoostingClassifier(
        random_state=RANDOM_STATE,
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
    )
    gb.fit(X_train, y_train)
    return gb


def _compute_feature_importances(
    model,
) -> np.ndarray | None:
    """
    Compute a generic feature importance vector for the given model.

    For tree-based models, uses feature_importances_.
    For linear models, uses the L2 norm of coefficients across classes.
    Returns None if not available.
    """
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        return np.linalg.norm(coef, axis=0)
    return None


def run_training(cfg: TrainConfig | None = None) -> None:
    """Main training routine."""
    if cfg is None:
        cfg = TrainConfig()

    logger.info("Loading processed training data...")
    df = load_processed_train()
    logger.info("Data shape: %s", df.shape)

    X, y, feature_names = _prepare_features_and_labels(df)

    # Global means of each feature (used for simple explanations)
    feature_means = X.mean(axis=0)

    # Decide whether to use stratified split (only if all classes have >= 2 samples)
    class_counts = np.bincount(y)
    min_count = class_counts.min()
    unique_classes = class_counts.size

    if unique_classes > 1 and min_count >= 2:
        stratify_arg = y
        logger.info(
            "Using stratified train/test split (class counts: %s).",
            class_counts.tolist(),
        )
    else:
        stratify_arg = None
        logger.warning(
            "Not using stratified split due to small sample per class. "
            "Class counts: %s",
            class_counts.tolist(),
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify_arg,
    )

    logger.info("Training baseline Logistic Regression...")
    log_reg, scaler = _train_log_reg(X_train, y_train)
    X_val_scaled = scaler.transform(X_val)
    y_val_pred_lr = log_reg.predict(X_val_scaled)
    y_val_proba_lr = log_reg.predict_proba(X_val_scaled)
    metrics_lr = compute_classification_metrics(
        y_val, y_val_pred_lr, y_val_proba_lr, CLASS_LABELS
    )
    logger.info("Logistic Regression metrics: %s", metrics_lr)

    logger.info("Training Gradient Boosting model...")
    gb = _train_gb(X_train, y_train)
    y_val_pred_gb = gb.predict(X_val)
    y_val_proba_gb = gb.predict_proba(X_val)
    metrics_gb = compute_classification_metrics(
        y_val, y_val_pred_gb, y_val_proba_gb, CLASS_LABELS
    )
    logger.info("Gradient Boosting metrics: %s", metrics_gb)

    # Choose best model based on log_loss (lower is better)
    if metrics_gb["log_loss"] <= metrics_lr["log_loss"]:
        best_model = gb
        model_type = "gradient_boosting"
        scaler_to_save = None  # not needed; GB doesn't need scaling
        logger.info("Selected Gradient Boosting as best model.")
    else:
        best_model = log_reg
        model_type = "logistic_regression"
        scaler_to_save = scaler
        logger.info("Selected Logistic Regression as best model.")

    feature_importances = _compute_feature_importances(best_model)
    if feature_importances is not None:
        # Normalize for nicer interpretation
        total = feature_importances.sum() + 1e-12
        feature_importances = feature_importances / total

    model_artifact = {
        "model_type": model_type,
        "model": best_model,
        "scaler": scaler_to_save,
        "feature_names": feature_names,
        "class_labels": CLASS_LABELS,
        "feature_means": feature_means,
        "feature_importances": feature_importances,
    }

    out_path = get_model_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, out_path)
    logger.info("Saved best model (%s) to %s", model_type, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FootyMind models.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Validation set fraction (default from TrainConfig).",
    )
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.test_size is not None:
        cfg.test_size = args.test_size

    run_training(cfg)


if __name__ == "__main__":
    main()
