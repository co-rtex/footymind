"""
End-to-end ETL pipeline for FootyMind.

Usage (from project root, with the virtualenv activated):

    python -m footymind.features.etl_pipeline

This will:
- Load raw data from data/raw/sample_matches.csv (or config default).
- Build rolling team form features and outcome labels.
- Save the processed training dataset to data/processed/train.csv.
"""

from __future__ import annotations

import argparse

from footymind.config import PROCESSED_DATA_DIR
from footymind.data.data_loader import load_raw_matches
from footymind.features.feature_builder import FeatureConfig, build_match_features
from footymind.utils.logging_utils import get_logger
from footymind.utils.paths import get_processed_data_path

logger = get_logger(__name__)


def run_etl(recent_form_window: int | None = None) -> None:
    """Run the ETL pipeline with the given parameters."""
    if recent_form_window is not None:
        cfg = FeatureConfig(recent_form_window=recent_form_window)
    else:
        cfg = FeatureConfig()

    logger.info("Starting ETL pipeline...")
    df_raw = load_raw_matches()
    features_df, labeled_matches_df = build_match_features(df_raw, cfg)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = get_processed_data_path("train.csv")
    features_df.to_csv(out_path, index=False)

    logger.info(
        "ETL complete. Saved processed training data with %d rows to %s",
        len(features_df),
        out_path,
    )

    matches_out_path = get_processed_data_path("matches_with_features.csv")
    labeled_matches_df.to_csv(matches_out_path, index=False)
    logger.info(
        "Saved enriched matches with features to %s", matches_out_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FootyMind ETL pipeline.")
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Recent form window size (number of previous matches per team). "
        "If not provided, uses the default from config.py.",
    )
    args = parser.parse_args()
    run_etl(recent_form_window=args.window)


if __name__ == "__main__":
    main()
