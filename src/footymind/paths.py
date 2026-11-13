"""Filesystem layout. Override the data root with ``FOOTYMIND_DATA_DIR``."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FOOTYMIND_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
