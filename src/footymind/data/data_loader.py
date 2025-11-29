"""
Data loading utilities for FootyMind.

This module provides functions to load raw Premier League match data and
processed feature datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from footymind.config import PROCESSED_DATA_DIR
from footymind.data.schema import validate_raw_matches_df
from footymind.utils.logging_utils import get_logger
from footymind.utils.paths import (
    get_raw_data_path,
    get_processed_data_path,
)

logger = get_logger(__name__)


def load_raw_matches(path: Optional[Path | str] = None) -> pd.DataFrame:
    """
    Load raw match data from a CSV file and validate it.

    Parameters
    ----------
    path : pathlib.Path | str | None
        Path to the raw CSV file. If None, uses the default path from config.

    Returns
    -------
    pandas.DataFrame
        Validated raw matches DataFrame.
    """
    csv_path = Path(path) if path is not None else get_raw_data_path()
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {csv_path}")

    logger.info("Loading raw match data from %s", csv_path)
    df = pd.read_csv(csv_path)
    df = validate_raw_matches_df(df)
    logger.info("Loaded %d valid raw match rows.", len(df))
    return df


def load_processed_train(path: Optional[Path | str] = None) -> pd.DataFrame:
    """
    Load processed training data (features + labels).

    Parameters
    ----------
    path : pathlib.Path | str | None
        Path to the processed CSV file. If None, uses 'train.csv' in the
        processed data directory.

    Returns
    -------
    pandas.DataFrame
        Processed training DataFrame.

    Raises
    ------
    FileNotFoundError
        If the processed file does not exist.
    """
    if path is None:
        csv_path = get_processed_data_path("train.csv")
    else:
        csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed training data not found: {csv_path}. "
            f"Run the ETL pipeline first."
        )

    logger.info("Loading processed training data from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d processed training rows.", len(df))
    return df


def list_processed_files() -> list[Path]:
    """
    List all processed CSV files in the processed data directory.

    Returns
    -------
    List[pathlib.Path]
        List of paths to processed CSV files.
    """
    return sorted(PROCESSED_DATA_DIR.glob("*.csv"))
