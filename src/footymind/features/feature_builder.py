# path: src/footymind/features/feature_builder.py
"""
Feature engineering utilities for FootyMind.

This module transforms raw match data into model-ready features:

- Creates outcome labels (home_win / draw / away_win).
- Builds team-centric long-format data with per-match stats from each team's
  perspective.
- Computes rolling "recent form" features for each team based on their last
  N matches prior to the current one.
- Merges home and away team features back into a match-level dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from footymind.config import (
    RECENT_FORM_WINDOW,
    MIN_MATCHES_FOR_FEATURES,
    TARGET_COLUMN,
    CLASS_LABELS,
)
from footymind.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    recent_form_window: int = RECENT_FORM_WINDOW
    min_matches_for_features: int = MIN_MATCHES_FOR_FEATURES


def compute_outcome_label(
    home_goals: int,
    away_goals: int,
) -> str:
    """
    Compute the match outcome from the home team's perspective.

    Parameters
    ----------
    home_goals : int
        Goals scored by the home team.
    away_goals : int
        Goals scored by the away team.

    Returns
    -------
    str
        One of 'home_win', 'draw', 'away_win'.
    """
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def add_outcome_labels(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Add the outcome label column to the raw matches DataFrame.

    Parameters
    ----------
    df_raw : pandas.DataFrame
        Raw matches DataFrame with home_goals and away_goals.

    Returns
    -------
    pandas.DataFrame
        DataFrame with an additional TARGET_COLUMN label.
    """
    df = df_raw.copy()
    df[TARGET_COLUMN] = [
        compute_outcome_label(h, a)
        for h, a in zip(df["home_goals"], df["away_goals"], strict=True)
    ]

    invalid_labels = set(df[TARGET_COLUMN].unique()) - set(CLASS_LABELS)
    if invalid_labels:
        raise ValueError(f"Unexpected outcome labels found: {invalid_labels}")

    return df


def _build_long_team_view(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a long-format DataFrame with one row per team per match.

    For each match, we create:
    - one row for the home team, is_home=1
    - one row for the away team, is_home=0

    Columns (per team row):
    - match_id, date, season, team, opponent, is_home
    - goals_for, goals_against
    - xG_for, xG_against
    - possession, shots, shots_on_target
    - corners, yellow_cards, red_cards
    - result (win/draw/loss from team perspective)
    - points (3/1/0)
    """
    df = df_matches.copy()

    # Ensure we have a unique match_id for each match
    if "match_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["match_id"] = df.index

    common_cols = ["match_id", "date", "season"]

    home_cols = common_cols + [
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

    df_home = df[home_cols].copy()
    df_home.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_goals": "goals_for",
            "away_goals": "goals_against",
            "home_xG": "xG_for",
            "away_xG": "xG_against",
            "home_possession": "possession",
            "away_possession": "opponent_possession",
            "home_shots": "shots",
            "away_shots": "opponent_shots",
            "home_shots_on_target": "shots_on_target",
            "away_shots_on_target": "opponent_shots_on_target",
            "home_corners": "corners",
            "away_corners": "opponent_corners",
            "home_yellow_cards": "yellow_cards",
            "away_yellow_cards": "opponent_yellow_cards",
            "home_red_cards": "red_cards",
            "away_red_cards": "opponent_red_cards",
        },
        inplace=True,
    )
    df_home["is_home"] = 1

    away_cols = home_cols  # same set
    df_away = df[away_cols].copy()
    df_away.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_goals": "goals_for",
            "home_goals": "goals_against",
            "away_xG": "xG_for",
            "home_xG": "xG_against",
            "away_possession": "possession",
            "home_possession": "opponent_possession",
            "away_shots": "shots",
            "home_shots": "opponent_shots",
            "away_shots_on_target": "shots_on_target",
            "home_shots_on_target": "opponent_shots_on_target",
            "away_corners": "corners",
            "home_corners": "opponent_corners",
            "away_yellow_cards": "yellow_cards",
            "home_yellow_cards": "opponent_yellow_cards",
            "away_red_cards": "red_cards",
            "home_red_cards": "opponent_red_cards",
        },
        inplace=True,
    )
    df_away["is_home"] = 0

    df_long = pd.concat([df_home, df_away], ignore_index=True)

    # Compute result and points from each team's perspective
    def _result(row) -> str:
        if row["goals_for"] > row["goals_against"]:
            return "win"
        if row["goals_for"] < row["goals_against"]:
            return "loss"
        return "draw"

    df_long["result"] = df_long.apply(_result, axis=1)
    df_long["points"] = df_long["result"].map({"win": 3, "draw": 1, "loss": 0})

    # Ensure date is datetime
    df_long["date"] = pd.to_datetime(df_long["date"], errors="coerce")

    return df_long


def _compute_rolling_features_for_team(
    df_long: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Compute rolling features for each team based on their previous matches.

    Key point: we use shift(1) before rolling to ensure that we only use data
    from matches that occurred BEFORE the current match (no leakage).
    """
    df = df_long.copy()
    df.sort_values(["team", "date", "match_id"], inplace=True)

    group = df.groupby("team", group_keys=False)

    def _apply_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        # How many matches has the team already played before this match?
        g["matches_played_before"] = np.arange(len(g))

        window = config.recent_form_window

        # Helper to shift then rolling
        def roll(col: str) -> pd.Series:
            return g[col].shift(1).rolling(window=window, min_periods=1).mean()

        g["rolling_goals_for_mean"] = roll("goals_for")
        g["rolling_goals_against_mean"] = roll("goals_against")
        g["rolling_goal_diff_mean"] = roll("goals_for") - roll("goals_against")

        g["rolling_xG_for_mean"] = roll("xG_for")
        g["rolling_xG_against_mean"] = roll("xG_against")

        g["rolling_possession_mean"] = roll("possession")
        g["rolling_shots_mean"] = roll("shots")
        g["rolling_shots_on_target_mean"] = roll("shots_on_target")
        g["rolling_corners_mean"] = roll("corners")
        g["rolling_yellow_mean"] = roll("yellow_cards")
        g["rolling_red_mean"] = roll("red_cards")
        g["rolling_points_mean"] = roll("points")

        return g

    df_with_roll = group.apply(_apply_group)

    return df_with_roll


def build_match_features(
    df_raw: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build match-level features and labels from raw data.

    Parameters
    ----------
    df_raw : pandas.DataFrame
        Raw matches DataFrame (already validated).
    config : FeatureConfig | None
        Feature configuration. If None, uses defaults from config.py.

    Returns
    -------
    (features_df, labeled_matches_df)
        features_df : DataFrame with engineered features and outcome label.
        labeled_matches_df : Original matches with added label column and
            engineered features (useful for API/analysis).
    """
    if config is None:
        config = FeatureConfig()

    logger.info(
        "Building match features with window=%d, min_matches_for_features=%d",
        config.recent_form_window,
        config.min_matches_for_features,
    )

    # 1) Add outcome labels
    df_labeled = add_outcome_labels(df_raw)

    # 2) Long team-view with rolling features
    df_long = _build_long_team_view(df_labeled)
    df_long_with_roll = _compute_rolling_features_for_team(df_long, config)

    # 3) Filter rows where team has played enough matches before
    df_long_with_roll = df_long_with_roll[
        df_long_with_roll["matches_played_before"] >= config.min_matches_for_features
    ]

    # 4) Split into home and away feature sets keyed by match_id
    home_long = df_long_with_roll[df_long_with_roll["is_home"] == 1].copy()
    away_long = df_long_with_roll[df_long_with_roll["is_home"] == 0].copy()

    # These are the engineered numeric features we want to use
    base_stats = [
        "matches_played_before",
        "rolling_goals_for_mean",
        "rolling_goals_against_mean",
        "rolling_goal_diff_mean",
        "rolling_xG_for_mean",
        "rolling_xG_against_mean",
        "rolling_possession_mean",
        "rolling_shots_mean",
        "rolling_shots_on_target_mean",
        "rolling_corners_mean",
        "rolling_yellow_mean",
        "rolling_red_mean",
        "rolling_points_mean",
    ]

    # Home features: one row per match_id with prefixed column names
    home_feats = home_long[["match_id"] + base_stats].copy()
    home_feats = home_feats.add_prefix("home_")
    home_feats.rename(columns={"home_match_id": "match_id"}, inplace=True)

    # Away features: one row per match_id with prefixed column names
    away_feats = away_long[["match_id"] + base_stats].copy()
    away_feats = away_feats.add_prefix("away_")
    away_feats.rename(columns={"away_match_id": "match_id"}, inplace=True)

    # 5) Merge with labeled matches (which still has home_team, away_team, etc.)
    df_matches = df_labeled.copy()
    if "match_id" not in df_matches.columns:
        df_matches = df_matches.reset_index(drop=True)
        df_matches["match_id"] = df_matches.index

    merged = (
        df_matches.merge(home_feats, on="match_id", how="inner")
        .merge(away_feats, on="match_id", how="inner")
    )

    # 6) Build final features DataFrame:
    # Only the engineered numeric rolling features + label
    engineered_columns: List[str] = (
        [f"home_{col}" for col in base_stats]
        + [f"away_{col}" for col in base_stats]
    )

    missing_cols = [c for c in engineered_columns if c not in merged.columns]
    if missing_cols:
        raise ValueError(
            f"Expected engineered feature columns missing: {missing_cols}")

    features_df = merged[engineered_columns + [TARGET_COLUMN]].copy()

    logger.info(
        "Built features for %d matches (out of %d raw).",
        len(features_df),
        len(df_raw),
    )

    return features_df, merged
