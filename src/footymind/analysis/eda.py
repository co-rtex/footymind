"""
Exploratory Data Analysis (EDA) for FootyMind.

Usage (from project root):

    cd src
    python -m footymind.analysis.eda

This will:
- Load raw and processed data
- Print basic summaries to the console
- Save plots into the `plots/` directory:
    * plots/eda_outcome_distribution.png
    * plots/eda_feature_correlations.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from footymind.config import CLASS_LABELS, TARGET_COLUMN
from footymind.data.data_loader import load_processed_train, load_raw_matches
from footymind.utils.logging_utils import get_logger
from footymind.utils.paths import get_project_root

logger = get_logger(__name__)


def _ensure_plots_dir() -> Path:
    root = get_project_root()
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def _ensure_outcome_column(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure an 'outcome' column exists (home_win/draw/away_win) based on goals.
    Does not modify the input DataFrame in-place.
    """
    df = df_raw.copy()
    if "outcome" in df.columns:
        return df

    if "home_goals" not in df.columns or "away_goals" not in df.columns:
        logger.warning(
            "Raw data has no 'outcome' column and no goal columns; "
            "cannot derive outcomes."
        )
        return df

    def _label(row: pd.Series) -> str:
        if row["home_goals"] > row["away_goals"]:
            return "home_win"
        if row["home_goals"] < row["away_goals"]:
            return "away_win"
        return "draw"

    df["outcome"] = df.apply(_label, axis=1)
    return df


def plot_outcome_distribution(df_raw: pd.DataFrame, plots_dir: Path) -> None:
    """
    Plot the distribution of match outcomes from the raw dataset.
    """
    df = _ensure_outcome_column(df_raw)

    if "outcome" not in df.columns:
        logger.warning(
            "Skipping outcome distribution plot (no outcome column).")
        return

    counts = df["outcome"].value_counts().reindex(CLASS_LABELS, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title("Match outcome distribution (raw data)")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.05, str(v), ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = plots_dir / "eda_outcome_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved outcome distribution plot to %s", path)


def plot_feature_correlations(
    df_processed: pd.DataFrame, plots_dir: Path, max_features: int = 15
) -> None:
    """
    Plot a correlation heatmap for numeric features in the processed dataset.

    For readability, we limit to the first `max_features` numeric features
    (excluding the label column).
    """
    numeric_cols = df_processed.select_dtypes(
        include=["number"]).columns.tolist()
    feature_cols: List[str] = [c for c in numeric_cols if c != TARGET_COLUMN]

    if not feature_cols:
        logger.warning("No numeric feature columns found in processed data.")
        return

    if len(feature_cols) > max_features:
        feature_cols = feature_cols[:max_features]
        logger.info(
            "Limiting correlation heatmap to first %d numeric features.",
            max_features,
        )

    corr = df_processed[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(1 + 0.5 * len(feature_cols), 6))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        center=0.0,
        square=True,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Feature correlation heatmap (subset)")
    fig.tight_layout()

    path = plots_dir / "eda_feature_correlations.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved feature correlation heatmap to %s", path)


def main() -> None:
    plots_dir = _ensure_plots_dir()

    # Raw data summary
    logger.info("Loading raw matches for EDA...")
    df_raw = load_raw_matches()
    logger.info("Raw data shape: %s", df_raw.shape)
    logger.info("Raw columns: %s", df_raw.columns.tolist())

    # Basic stats
    if {"home_goals", "away_goals"}.issubset(df_raw.columns):
        df_raw["total_goals"] = df_raw["home_goals"] + df_raw["away_goals"]
        logger.info(
            "Average total goals per match: %.2f",
            df_raw["total_goals"].mean(),
        )

    # Processed data summary
    logger.info("Loading processed training data for EDA...")
    try:
        df_processed = load_processed_train()
        logger.info("Processed data shape: %s", df_processed.shape)
        logger.info(
            "Processed numeric feature columns (first 10): %s",
            df_processed.select_dtypes(include=["number"])
            .columns.tolist()[:10],
        )
    except FileNotFoundError:
        logger.error(
            "Processed training data not found. Run ETL first:\n"
            "  cd src\n"
            "  python -m footymind.features.etl_pipeline"
        )
        return

    # Plots
    plot_outcome_distribution(df_raw, plots_dir)
    plot_feature_correlations(df_processed, plots_dir)

    logger.info("EDA complete.")


if __name__ == "__main__":
    main()
