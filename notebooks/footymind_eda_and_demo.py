# %%
"""
FootyMind EDA & Demo Script (Notebook-Style)

This file is written as a Python script with notebook-style cells using `# %%`.
You can:

- Open it in VS Code and run cells interactively, OR
- Run it as a normal script:

    cd /Users/christiancortez/Desktop/footymind
    source .venv/bin/activate
    python notebooks/footymind_eda_and_demo.py
"""

# %%
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path so we can import the footymind package
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../footymind
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from footymind.config import CLASS_LABELS, TARGET_COLUMN  # noqa: E402
from footymind.data.data_loader import (  # noqa: E402
    load_processed_train,
    load_raw_matches,
)
from footymind.utils.paths import get_model_path  # noqa: E402

# %%
# Basic info
print("Project root:", PROJECT_ROOT)
print("Using src at:", SRC_ROOT)

# %%
# 1. Load raw data
print("\n=== 1. Load raw match data ===")
df_raw = load_raw_matches()
print("Raw data shape:", df_raw.shape)
print("Raw columns:", df_raw.columns.tolist())
print("\nRaw head():")
print(df_raw.head())

# Add outcome if missing (home_win/draw/away_win)
if "outcome" not in df_raw.columns and {
    "home_goals",
    "away_goals",
}.issubset(df_raw.columns):
    def _label(row: pd.Series) -> str:
        if row["home_goals"] > row["away_goals"]:
            return "home_win"
        if row["home_goals"] < row["away_goals"]:
            return "away_win"
        return "draw"

    df_raw["outcome"] = df_raw.apply(_label, axis=1)

# %%
# 2. Outcome distribution (raw)
print("\n=== 2. Outcome distribution (raw data) ===")
if "outcome" in df_raw.columns:
    outcome_counts = df_raw["outcome"].value_counts().reindex(
        CLASS_LABELS, fill_value=0
    )
    print(outcome_counts)

    # Quick bar plot
    plt.figure(figsize=(6, 4))
    sns.barplot(x=outcome_counts.index, y=outcome_counts.values)
    plt.title("Match outcome distribution (raw data)")
    plt.xlabel("Outcome")
    plt.ylabel("Count")
    for i, v in enumerate(outcome_counts.values):
        plt.text(i, v + 0.05, str(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()
else:
    print("No 'outcome' column available in raw data; skipping this step.")

# %%
# 3. Load processed training data
print("\n=== 3. Load processed training data ===")
try:
    df_processed = load_processed_train()
except FileNotFoundError:
    print(
        "Processed training data not found. Run the ETL pipeline first:\n"
        "  cd src\n"
        "  python -m footymind.features.etl_pipeline"
    )
    raise

print("Processed shape:", df_processed.shape)
print("Processed columns (first 15):", df_processed.columns.tolist()[:15])

numeric_cols = df_processed.select_dtypes(include=["number"]).columns.tolist()
feature_cols = [c for c in numeric_cols if c != TARGET_COLUMN]

print("\nNumeric feature columns (first 10):", feature_cols[:10])

# %%
# 4. Simple correlation heatmap for a subset of features
print("\n=== 4. Feature correlation (subset) ===")
if feature_cols:
    subset = feature_cols[: min(12, len(feature_cols))]
    corr = df_processed[subset].corr()
    print("Using feature subset for correlation:", subset)

    plt.figure(figsize=(1 + 0.5 * len(subset), 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0.0, square=True)
    plt.title("Feature correlation heatmap (subset)")
    plt.tight_layout()
    plt.show()
else:
    print("No numeric feature columns found; skipping correlation heatmap.")

# %%
# 5. Load trained model artifact
print("\n=== 5. Load trained model artifact ===")
model_path = get_model_path()
print("Model path:", model_path)

try:
    artifact = joblib.load(model_path)
except FileNotFoundError:
    print(
        "Model artifact not found. Train a model first:\n"
        "  cd src\n"
        "  python -m footymind.models.train_model"
    )
    raise

model = artifact["model"]
scaler = artifact.get("scaler")
feature_names = artifact["feature_names"]
class_labels = artifact["class_labels"]
feature_means = artifact.get("feature_means")
feature_importances = artifact.get("feature_importances")

print("Loaded model type:", type(model).__name__)
print("Number of features:", len(feature_names))
print("Class labels:", class_labels)

# %%
# 6. Pick a match, build feature vector, and predict
print("\n=== 6. Demo: predict for a single match ===")

# Use processed matches_with_features to get a match_id and its features
matches_with_features_path = (
    PROJECT_ROOT / "data" / "processed" / "matches_with_features.csv"
)
if not matches_with_features_path.exists():
    print(
        "Could not find matches_with_features.csv. Run the ETL pipeline first:\n"
        "  cd src\n"
        "  python -m footymind.features.etl_pipeline"
    )
    raise SystemExit(1)

df_matches = pd.read_csv(matches_with_features_path)
print("Matches with features shape:", df_matches.shape)

# Just pick the last match as an example
row = df_matches.sort_values("date").iloc[-1]
match_id = int(row["match_id"])
home_team = row.get("home_team")
away_team = row.get("away_team")
date = row.get("date")
season = row.get("season")
actual_outcome = row.get("outcome")

print(
    f"Using match_id={match_id}: "
    f"{date} – {home_team} vs {away_team} ({season}), "
    f"actual outcome={actual_outcome}"
)

# Build feature vector
x_values = row[feature_names].astype(float).values
X = np.asarray([x_values], dtype=float)

if scaler is not None:
    X_input = scaler.transform(X)
else:
    X_input = X

proba = model.predict_proba(X_input)[0]
pred_idx = int(np.argmax(proba))
pred_label = class_labels[pred_idx]

print("\nPredicted probabilities:")
for label, p in zip(class_labels, proba):
    print(f"  {label:9s}: {p:.3f}")

print(f"\nPredicted outcome: {pred_label}")

# %%
# 7. Simple explanation using feature importances & deviation from mean
print("\n=== 7. Simple explanation ===")

if feature_means is not None and feature_importances is not None:
    values = x_values.astype(float)
    means = np.asarray(feature_means, dtype=float)
    imps = np.asarray(feature_importances, dtype=float)

    if (
        values.shape[0] == means.shape[0]
        and values.shape[0] == imps.shape[0]
    ):
        contributions = (values - means) * imps
        idx_sorted = np.argsort(np.abs(contributions))[::-1]
        top_k = min(10, len(idx_sorted))
        print(f"Top {top_k} contributing features (by abs contribution):\n")
        for i in idx_sorted[:top_k]:
            print(
                f"- {feature_names[i]:30s} "
                f"value={values[i]:8.3f}, "
                f"mean={means[i]:8.3f}, "
                f"importance={imps[i]:8.3f}, "
                f"contrib={contributions[i]:8.3f}"
            )
    else:
        print("Feature shapes do not match; cannot compute contributions.")
else:
    print("No feature_means or feature_importances in artifact; "
          "run the newer training script or update the artifact.")

# %%
print("\nDone. You can re-run cells above interactively in VS Code or Jupyter.")
