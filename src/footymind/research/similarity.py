"""Similar-moment retrieval for pass decisions (session 3).

For a pass decision, find the ``k`` most similar decision-time situations from **other
matches of the same provider** and report what actually happened in them. The comparison
set is observed outcomes, not a model output, so a user can open and check every neighbour.
Similarity is Euclidean distance on standardised decision-time features (the RQ2 primary
feature set, ``SIMILARITY_FEATURES``). Standardisation statistics come from the index rows
and missing values are imputed with the index median.

Honesty constraints:

* neighbours never come from the query's own match (no near-duplicate frames of the same
  passage of play), and never from another provider;
* the neighbours' completion rate is an empirical *comparison*, not a prediction for this
  pass; ``research/similarity_validation.py`` measures how informative it is held-out.

Index files live in ``data/processed/similarity_<provider>.parquet`` (derived, not
committed) and are built with ``footymind build-similarity-index``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from footymind.paths import PROCESSED_DIR
from footymind.research.pass_dataset import (
    DECISION_SPATIAL_FEATURES,
    TEMPORAL_EVENT_FEATURES,
    TEMPORAL_TRACK_FEATURES,
    TIME_ON_BALL_FEATURES,
    DatasetConfig,
)

SIMILARITY_FEATURES = [
    *DECISION_SPATIAL_FEATURES,
    *TEMPORAL_EVENT_FEATURES,
    *TIME_ON_BALL_FEATURES,
    *TEMPORAL_TRACK_FEATURES,
]
# Context columns kept for display. Label-side columns are shown, never used for similarity.
INDEX_COLUMNS = [
    "match_id",
    "event_id",
    "possession_id",
    "frame",
    "team",
    "passer_id",
    "success",
    "ball_x_att",
    "press_nearest_def_m",
    "passer_control_s",
]

# every column stored in an index file, without duplicates
STORED_COLUMNS = [*INDEX_COLUMNS, *(f for f in SIMILARITY_FEATURES if f not in INDEX_COLUMNS)]


# Every similarity path (index, serving and validation) reads the situation 0.2 s BEFORE the
# release frame: on SkillCorner the start frame often already contains the kick (RQ2 review).
GUARD_S = 0.2


def similarity_dataset_config() -> DatasetConfig:
    return DatasetConfig(static_offset_s=GUARD_S, temporal_offset_s=GUARD_S)


def index_path(provider: str) -> Path:
    return PROCESSED_DIR / f"similarity_{provider}.parquet"


@dataclass
class SimilarityIndex:
    rows: pl.DataFrame  # INDEX_COLUMNS + SIMILARITY_FEATURES
    mean: np.ndarray
    scale: np.ndarray
    median: np.ndarray
    z: np.ndarray  # (n, d) standardised, imputed

    @classmethod
    def from_rows(cls, rows: pl.DataFrame) -> SimilarityIndex:
        x = rows.select(SIMILARITY_FEATURES).to_numpy().astype(float)
        median = np.nanmedian(x, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        xi = np.where(np.isfinite(x), x, median[None, :])
        mean = xi.mean(axis=0)
        scale = xi.std(axis=0)
        scale = np.where(scale > 1e-9, scale, 1.0)
        return cls(rows, mean, scale, median, (xi - mean) / scale)

    def standardise(self, x: np.ndarray) -> np.ndarray:
        xi = np.where(np.isfinite(x), x, self.median)
        return np.asarray((xi - self.mean) / self.scale)

    def query(
        self, x: np.ndarray, exclude_match: str | None, k: int = 10
    ) -> tuple[pl.DataFrame, np.ndarray]:
        """The ``k`` nearest index rows to feature vector ``x`` (not from ``exclude_match``)."""
        d = np.linalg.norm(self.z - self.standardise(x)[None, :], axis=1)
        if exclude_match is not None:
            d = np.where(self.rows["match_id"].to_numpy() == exclude_match, np.inf, d)
        k = int(min(k, np.isfinite(d).sum()))
        idx = np.argsort(d)[:k]
        return self.rows[idx.tolist()], d[idx]


def build_index(provider: str, match_ids: list[str]) -> Path:
    """Build and save the provider's index from the research dataset rows (rules v2)."""
    from footymind.research.pass_dataset import build

    if any(not m.startswith(provider + "-") for m in match_ids):
        raise ValueError("an index must contain a single provider's matches")
    df = build(match_ids, similarity_dataset_config())
    out = index_path(provider)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.select(STORED_COLUMNS).write_parquet(out)
    return out


_CACHE: dict[str, SimilarityIndex] = {}


def load_index(provider: str) -> SimilarityIndex | None:
    if provider not in _CACHE:
        p = index_path(provider)
        if not p.exists():
            return None
        _CACHE[provider] = SimilarityIndex.from_rows(pl.read_parquet(p))
    return _CACHE[provider]


NEXT_DECISION_WINDOW_S = 2.0


def decision_row(match_id: str, frame: int) -> dict[str, object] | None:
    """Feature row of the next pass decision at or after ``frame`` (within 2 s), or None.

    It uses exactly the research dataset's feature builder (train/serve consistency). None
    means there is no labelled pass attempt starting in the window, or its features cannot
    be computed (for example, an unobserved ball).
    """
    from footymind.data.store import load_match
    from footymind.research.pass_dataset import (
        _features_for,
        _prev_event_features,
        label_events,
    )

    b = load_match(match_id)
    td = b.tracking
    window = round(NEXT_DECISION_WINDOW_S * td.fps)
    cfg = similarity_dataset_config()
    ev = b.events.filter(
        (pl.col("start_frame") >= frame) & (pl.col("start_frame") <= frame + window)
    )
    if ev.is_empty():  # cheap early exit before labelling the whole match
        return None
    # Event history needs the whole possession, so label the full match and then filter.
    cand = label_events(_prev_event_features(b.events), cfg).filter(
        (pl.col("start_frame") >= frame) & (pl.col("start_frame") <= frame + window)
    )
    for r in cand.sort("start_frame").iter_rows(named=True):
        row = _features_for(match_id, r, td, cfg)
        if row is not None:
            return row
    return None
