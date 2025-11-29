# path: src/footymind/api/main.py
"""
FastAPI app exposing FootyMind prediction endpoints.

Endpoints:
- GET  /health                 -> simple health check
- GET  /matches                -> list of matches with basic info
- POST /predict_by_match_id    -> predict outcome + explanation for a given match_id
"""

from __future__ import annotations

from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from footymind.utils.logging_utils import get_logger
from footymind.utils.paths import get_model_path, get_processed_data_path

logger = get_logger(__name__)

app = FastAPI(
    title="FootyMind API",
    version="0.1.0",
    description="Premier League match outcome predictor",
)

# Global state populated at startup
MODEL_ARTIFACT: Dict[str, Any] | None = None
MATCHES_DF: pd.DataFrame | None = None


class PredictByMatchIdRequest(BaseModel):
    match_id: int


def _load_model_artifact() -> Dict[str, Any]:
    """Load the trained model artifact from disk."""
    global MODEL_ARTIFACT
    if MODEL_ARTIFACT is not None:
        return MODEL_ARTIFACT

    model_path = get_model_path()
    if not model_path.exists():
        raise RuntimeError(
            f"Model file not found at {model_path}. "
            "Have you run `python -m footymind.models.train_model`?"
        )

    MODEL_ARTIFACT = joblib.load(model_path)
    logger.info("Loaded model artifact from %s", model_path)
    return MODEL_ARTIFACT


def _load_matches_with_features() -> pd.DataFrame:
    """Load matches_with_features.csv into a DataFrame."""
    global MATCHES_DF
    if MATCHES_DF is not None:
        return MATCHES_DF

    path = get_processed_data_path("matches_with_features.csv")
    if not path.exists():
        raise RuntimeError(
            f"Processed matches file not found at {path}. "
            "Have you run `python -m footymind.features.etl_pipeline`?"
        )

    MATCHES_DF = pd.read_csv(path)
    logger.info(
        "Loaded %d matches with features from %s", len(MATCHES_DF), path
    )
    return MATCHES_DF


@app.on_event("startup")
def startup_event() -> None:
    """Load model and data at application startup."""
    try:
        _load_model_artifact()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load model artifact on startup: %s", exc)

    try:
        _load_matches_with_features()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to load matches_with_features on startup: %s", exc)


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    # We keep it minimal for now since your scripts already expect {"status": "ok"}
    return {"status": "ok"}


@app.get("/matches")
def list_matches() -> List[Dict[str, Any]]:
    """
    Return a list of matches with basic info.

    Each item:
    - match_id
    - date
    - season
    - home_team
    - away_team
    - outcome
    """
    df = _load_matches_with_features()

    required_cols = [
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "outcome",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Expected columns missing in matches_with_features.csv: {missing}",
        )

    records = df[required_cols].to_dict(orient="records")
    return records


def _build_explanation(
    feature_names: List[str],
    values: np.ndarray,
    feature_means: np.ndarray | None,
    feature_importances: np.ndarray | None,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Build a simple explanation by combining feature importance with deviation
    from mean:

        contribution_i = (value_i - mean_i) * importance_i

    Returns a list of up to top_k features, sorted by absolute contribution.
    """
    if feature_means is None or feature_importances is None:
        return []

    # Ensure everything is numpy arrays with matching shapes
    values = np.asarray(values, dtype=float)
    feature_means = np.asarray(feature_means, dtype=float)
    feature_importances = np.asarray(feature_importances, dtype=float)

    if (
        values.shape[0] != feature_means.shape[0]
        or values.shape[0] != feature_importances.shape[0]
    ):
        logger.warning(
            "Shape mismatch in explanation: values=%s, means=%s, imps=%s",
            values.shape,
            feature_means.shape,
            feature_importances.shape,
        )
        return []

    contributions = (values - feature_means) * feature_importances
    idx_sorted = np.argsort(np.abs(contributions))[::-1]
    top_k = min(top_k, len(idx_sorted))
    idx_top = idx_sorted[:top_k]

    explanation: List[Dict[str, Any]] = []
    for i in idx_top:
        contrib = float(contributions[i])
        explanation.append(
            {
                "feature_name": feature_names[i],
                "value": float(values[i]),
                "mean": float(feature_means[i]),
                "importance": float(feature_importances[i]),
                "contribution": contrib,
                "direction": "positive"
                if contrib > 0
                else "negative" if contrib < 0 else "neutral",
            }
        )

    return explanation


@app.post("/predict_by_match_id")
def predict_by_match_id(payload: PredictByMatchIdRequest) -> Dict[str, Any]:
    """
    Predict outcome and provide explanation for a given match_id.

    Request:
        { "match_id": <int> }

    Response:
        {
          "match_id": ...,
          "home_team": ...,
          "away_team": ...,
          "season": ...,
          "date": ...,
          "predicted_class": "home_win" | "draw" | "away_win",
          "class_probabilities": { "home_win": 0.7, "draw": 0.2, "away_win": 0.1 },
          "explanation": [
              {
                "feature_name": "...",
                "value": ...,
                "mean": ...,
                "importance": ...,
                "contribution": ...,
                "direction": "positive" | "negative" | "neutral"
              },
              ...
          ]
        }
    """
    artifact = _load_model_artifact()
    df = _load_matches_with_features()

    # Locate the match row
    mask = df["match_id"] == payload.match_id
    if not mask.any():
        raise HTTPException(
            status_code=404,
            detail=f"No match found with match_id={payload.match_id}",
        )

    row = df.loc[mask].iloc[0]

    model = artifact["model"]
    scaler = artifact.get("scaler")
    feature_names: List[str] = artifact["feature_names"]
    class_labels: List[str] = artifact["class_labels"]
    feature_means = artifact.get("feature_means")
    feature_importances = artifact.get("feature_importances")

    # Extract feature vector in the same order as during training
    try:
        x_values = row[feature_names].astype(float).values
    except KeyError as exc:  # noqa: BLE001
        missing = [c for c in feature_names if c not in df.columns]
        raise HTTPException(
            status_code=500,
            detail=f"Feature columns missing in matches_with_features.csv: {missing}",
        ) from exc

    X = np.asarray([x_values], dtype=float)

    if scaler is not None:
        X_input = scaler.transform(X)
    else:
        X_input = X

    # Predict probabilities and class
    proba = model.predict_proba(X_input)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = class_labels[pred_idx]

    prob_dict = {label: float(p) for label, p in zip(class_labels, proba)}

    explanation = _build_explanation(
        feature_names=feature_names,
        values=x_values,
        feature_means=feature_means,
        feature_importances=feature_importances,
        top_k=8,
    )

    result: Dict[str, Any] = {
        "match_id": int(row["match_id"]),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "season": row.get("season"),
        "date": str(row.get("date")),
        "predicted_class": pred_label,
        "class_probabilities": prob_dict,
        "explanation": explanation,
    }

    return result
