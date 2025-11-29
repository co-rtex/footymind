"""
Helper functions for file and directory paths used in FootyMind.
"""

from pathlib import Path
from typing import Union

from footymind.config import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_MATCHES_FILENAME,
    MODELS_DIR,
    DEFAULT_MODEL_FILENAME,
)


PathLike = Union[str, Path]


def get_project_root() -> Path:
    """Return the root directory of the FootyMind project."""
    return PROJECT_ROOT


def get_raw_data_path(filename: str | None = None) -> Path:
    """
    Return the path to a raw data file.

    Parameters
    ----------
    filename : str | None
        Specific filename, or None for the default raw matches CSV.

    Returns
    -------
    Path
        Full path to the raw data file.
    """
    if filename is None:
        filename = RAW_MATCHES_FILENAME
    return RAW_DATA_DIR / filename


def get_processed_data_path(filename: str = "train.csv") -> Path:
    """
    Return the path to a processed data file.

    Parameters
    ----------
    filename : str
        Filename within the processed data directory.

    Returns
    -------
    Path
        Full path to the processed data file.
    """
    return PROCESSED_DATA_DIR / filename


def get_model_path(filename: str | None = None) -> Path:
    """
    Return the path to a model artifact file.

    Parameters
    ----------
    filename : str | None
        Specific filename, or None for the default model file.

    Returns
    -------
    Path
        Full path to the model artifact.
    """
    if filename is None:
        filename = DEFAULT_MODEL_FILENAME
    return MODELS_DIR / filename
