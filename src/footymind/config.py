"""
Global configuration for the FootyMind project.

This module centralizes paths and key parameters (e.g., rolling window size),
so you can tweak them in one place.
"""

from pathlib import Path

# Project root = folder that contains "src", "data", "models", etc.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# Default raw dataset
RAW_MATCHES_FILENAME: str = "sample_matches.csv"

# Models directory
MODELS_DIR: Path = PROJECT_ROOT / "models"
DEFAULT_MODEL_FILENAME: str = "footymind_model.joblib"

# Feature engineering parameters
RECENT_FORM_WINDOW: int = 5  # number of previous matches to consider for form

# Minimum number of past matches a team must have played BEFORE the current
# match to compute reliable rolling stats.
# For small demo datasets, keep this low (e.g., 1).
MIN_MATCHES_FOR_FEATURES: int = 1

# Target and label mapping
TARGET_COLUMN: str = "outcome"  # label column name in processed data
CLASS_LABELS = ["home_win", "draw", "away_win"]

# Reproducibility
RANDOM_STATE: int = 42

# Plot output directory (for evaluation/explainability)
PLOTS_DIR: Path = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Ensure directories exist
for _dir in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
