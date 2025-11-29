# path: src/footymind/models/metrics.py
"""
Metrics utilities for FootyMind models.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

from footymind.config import CLASS_LABELS


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    labels: list[str] | None = None,
) -> Dict[str, Any]:
    """
    Compute a set of basic classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True class indices (0..n_classes-1).
    y_pred : np.ndarray
        Predicted class indices.
    y_proba : np.ndarray
        Predicted probabilities with shape (n_samples, n_pred_classes).
        Columns are assumed to correspond to the classes in the underlying
        model's .classes_ attribute.
    labels : list[str] | None
        List of class label strings (e.g., CLASS_LABELS). If provided,
        confusion_matrix will be returned with fixed size.

    Returns
    -------
    dict
        {
          "accuracy": float,
          "log_loss": float,
          "baseline_accuracy": float,
          "confusion_matrix": list[list[int]],
        }
    """
    if labels is None:
        labels = CLASS_LABELS

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Majority-class baseline accuracy
    counts = np.bincount(y_true) if len(y_true) > 0 else np.array([0])
    if counts.sum() > 0:
        baseline_acc = counts.max() / counts.sum()
    else:
        baseline_acc = float("nan")

    # Log loss: let sklearn infer labels from y_true / y_proba
    try:
        ll = log_loss(y_true, y_proba)
    except ValueError:
        # This can happen in extreme edge cases (e.g., only one class present)
        ll = float("nan")

    # Confusion matrix: fixed size based on CLASS_LABELS length
    num_classes = len(labels)
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
    )

    return {
        "accuracy": float(acc),
        "log_loss": float(ll),
        "baseline_accuracy": float(baseline_acc),
        "confusion_matrix": cm.tolist(),
    }
