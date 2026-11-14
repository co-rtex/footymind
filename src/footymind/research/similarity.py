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
committed) and are built with ``footymind build-similarity-index``. Each file records, in
its parquet metadata, the pipeline version and the ``created_utc`` of every bundle it was
built from. ``load_index`` refuses an index whose metadata no longer matches the processed
bundles (``StaleIndexError``), and its in-process cache follows the file's modification time
and the bundles' manifests, so a rebuild or a re-ingest is picked up without a restart.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from footymind import paths
from footymind.data import store
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
    return paths.PROCESSED_DIR / f"similarity_{provider}.parquet"


META_PREFIX = "footymind:"


class StaleIndexError(RuntimeError):
    """The index was built from other bundles (or another pipeline version) than are served."""


def rebuild_command(provider: str) -> str:
    return f"footymind build-similarity-index --provider {provider}"


@dataclass
class SimilarityIndex:
    rows: pl.DataFrame  # INDEX_COLUMNS + SIMILARITY_FEATURES
    mean: np.ndarray
    scale: np.ndarray
    median: np.ndarray
    z: np.ndarray  # (n, d) standardised, imputed
    # (mtime_ns, size) of the file it was read from. None: placed in the cache directly
    # (tests, embedding code), so it is not tied to a file and never reloaded.
    source: tuple[int, int, int] | None = None

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


def _created(match_id: str) -> str | None:
    """The bundle's creation stamp, or None if its manifest cannot be read."""
    try:
        return str(store.read_manifest(match_id)["created_utc"])
    except Exception:
        return None


def indexable_matches(provider: str) -> tuple[list[str], list[str]]:
    """The provider's servable bundles that load, and those that are servable but do not.

    Bundles the server would refuse (another pipeline version, an unreadable manifest) are in
    neither list: they need ``footymind ingest`` first.
    """
    ok: list[str] = []
    broken: list[str] = []
    for m in store.available_matches():
        if not m.startswith(provider + "-") or not store.is_servable(m):
            continue
        try:
            store.load_match(m)
            ok.append(m)
        except Exception:  # a damaged data file under a valid manifest
            broken.append(m)
    return ok, broken


def build_index(provider: str, match_ids: list[str], skipped: list[str] | None = None) -> Path:
    """Build and save the provider's index from the research dataset rows (rules v2).

    ``skipped`` are servable bundles that could not be read; they are recorded so that the
    index is reported stale once they are repaired (re-ingested), and not before.
    """
    from footymind.research.pass_dataset import build

    if any(not m.startswith(provider + "-") for m in [*match_ids, *(skipped or [])]):
        raise ValueError("an index must contain a single provider's matches")
    # recorded before reading the bundles: if one is re-ingested during the build, the index
    # keeps the older timestamp and is reported stale, never stored as fresh
    meta = index_metadata(provider, match_ids, skipped)
    df = build(match_ids, similarity_dataset_config())
    return write_index(df, provider, match_ids, meta)


def index_metadata(
    provider: str, match_ids: list[str], skipped: list[str] | None = None
) -> dict[str, str]:
    """What an index is built from: provider, pipeline version and each bundle's creation."""
    bundles = {m: str(store.read_manifest(m)["created_utc"]) for m in sorted(match_ids)}
    left_out = {m: _created(m) for m in sorted(skipped or [])}
    return {
        META_PREFIX + "provider": provider,
        META_PREFIX + "pipeline_version": store.PIPELINE_VERSION,
        META_PREFIX + "bundles": json.dumps(bundles, sort_keys=True),
        META_PREFIX + "skipped": json.dumps(left_out, sort_keys=True),
    }


def write_index(
    df: pl.DataFrame,
    provider: str,
    match_ids: list[str],
    meta: dict[str, str] | None = None,
) -> Path:
    out = index_path(provider)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    meta = meta if meta is not None else index_metadata(provider, match_ids)
    df.select(STORED_COLUMNS).write_parquet(tmp, metadata=meta)
    os.replace(tmp, out)  # atomic: a running server never reads a half-written index
    return out


def check_index_metadata(meta: dict[str, str], provider: str) -> None:
    """Raise ``StaleIndexError`` unless the index matches the processed bundles served now."""
    fix = f"rebuild it with `{rebuild_command(provider)}`"
    name = index_path(provider).name
    version = meta.get(META_PREFIX + "pipeline_version")
    if version is None:
        raise StaleIndexError(f"similarity index {name} has no build metadata; {fix}")
    if version != store.PIPELINE_VERSION:
        raise StaleIndexError(
            f"similarity index {name} was built by pipeline v{version}, but the served "
            f"bundles are v{store.PIPELINE_VERSION}; {fix}"
        )
    if meta.get(META_PREFIX + "provider") != provider:
        raise StaleIndexError(f"similarity index {name} is not a {provider} index; {fix}")
    built: dict[str, str] = json.loads(meta.get(META_PREFIX + "bundles", "{}"))
    skipped: dict[str, str | None] = json.loads(meta.get(META_PREFIX + "skipped", "{}"))
    # Only bundles the server serves count: one stale or unreadable bundle must not take the
    # index down for every other match of the provider.
    servable = [
        m
        for m in store.available_matches()
        if m.startswith(provider + "-") and store.is_servable(m)
    ]
    changed = sorted(m for m in servable if m in built and _created(m) != built[m])
    if changed:
        more = " and others" if len(changed) > 3 else ""
        raise StaleIndexError(
            f"similarity index {name} predates a re-ingest of {', '.join(changed[:3])}{more}; {fix}"
        )
    # a match added (or repaired) since the build would silently be missing from everyone's
    # neighbours; one left out at build time because it could not be read is not "new"
    missing = sorted(
        m for m in servable if m not in built and (m not in skipped or _created(m) != skipped[m])
    )
    if missing:
        more = " and others" if len(missing) > 3 else ""
        raise StaleIndexError(
            f"similarity index {name} does not include {', '.join(missing[:3])}{more}; {fix}"
        )


def _file_token(p: Path) -> tuple[int, int, int] | None:
    try:
        st = os.stat(p)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _bundles_token(provider: str) -> tuple[tuple[str, tuple[int, int, int] | None], ...]:
    return tuple(
        (m, store.bundle_token(m))
        for m in store.available_matches()
        if m.startswith(provider + "-")
    )


_CACHE: dict[str, SimilarityIndex] = {}
_VALIDATED: dict[str, object] = {}  # provider -> bundles token at the last successful check


def load_index(provider: str) -> SimilarityIndex | None:
    """The provider's index, or None if it has not been built.

    Raises ``StaleIndexError`` if it no longer matches the processed bundles, and the read
    error for an unreadable file. The cache is checked on every call (a few ``os.stat``):
    the file is re-read when it changes, and its metadata is re-checked when a bundle does.
    """
    cached = _CACHE.get(provider)
    if cached is not None and cached.source is None:
        return cached
    p = index_path(provider)
    ftok = _file_token(p)
    if ftok is None:
        _CACHE.pop(provider, None)
        return None
    btok = _bundles_token(provider)
    if cached is not None and cached.source == ftok and _VALIDATED.get(provider) == btok:
        return cached
    try:
        check_index_metadata(pl.read_parquet_metadata(p), provider)
        if cached is None or cached.source != ftok:
            cached = SimilarityIndex.from_rows(pl.read_parquet(p))
            cached.source = ftok
    except Exception:
        _CACHE.pop(provider, None)  # never keep serving an index that failed its check
        _VALIDATED.pop(provider, None)
        raise
    _CACHE[provider] = cached
    _VALIDATED[provider] = btok
    return cached


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
