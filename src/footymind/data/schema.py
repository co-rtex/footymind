"""
Schema and validation utilities for raw Premier League match data.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from footymind.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Expected columns in the raw dataset
RAW_MATCHES_COLUMNS: List[str] = [
    "date",
    "season",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "home_xG",
    "away_xG",
    "home_possession",
    "away_possession",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_yellow_cards",
    "away_yellow_cards",
    "home_red_cards",
    "away_red_cards",
]


def validate_raw_matches_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that a DataFrame conforms to the expected raw matches schema.

    Checks:
    - All required columns are present.
    - No duplicate rows (based on all columns).
    - Coerces date column to datetime.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw matches DataFrame.

    Returns
    -------
    pandas.DataFrame
        A validated (and possibly slightly adjusted) DataFrame.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing = [col for col in RAW_MATCHES_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw match columns: {missing}")

    # Ensure date column is datetime
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        logger.warning("Some rows have invalid 'date' values after parsing.")

    # Drop duplicate rows if any (warn but don't fail)
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if after < before:
        logger.info("Dropped %d duplicate raw rows.", before - after)

    return df


def get_raw_schema_description() -> Dict[str, str]:
    """
    Return a human-readable description of the raw matches schema.

    Returns
    -------
    Dict[str, str]
        Mapping from column name to description.
    """
    return {
        "date": "Match date (YYYY-MM-DD)",
        "season": "Season identifier (e.g., '2023-2024')",
        "home_team": "Name of home team",
        "away_team": "Name of away team",
        "home_goals": "Goals scored by home team",
        "away_goals": "Goals scored by away team",
        "home_xG": "Expected goals for home team",
        "away_xG": "Expected goals for away team",
        "home_possession": "Home team possession percentage",
        "away_possession": "Away team possession percentage",
        "home_shots": "Total shots for home team",
        "away_shots": "Total shots for away team",
        "home_shots_on_target": "Shots on target for home team",
        "away_shots_on_target": "Shots on target for away team",
        "home_corners": "Corners won by home team",
        "away_corners": "Corners won by away team",
        "home_yellow_cards": "Yellow cards received by home team",
        "away_yellow_cards": "Yellow cards received by away team",
        "home_red_cards": "Red cards received by home team",
        "away_red_cards": "Red cards received by away team",
    }
