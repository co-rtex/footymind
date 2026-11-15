"""Similar-moment index on SYNTHETIC feature rows (not match observations)."""

import numpy as np
import polars as pl
import pytest

from footymind.research.similarity import (
    INDEX_COLUMNS,
    SIMILARITY_FEATURES,
    SimilarityIndex,
    build_index,
)


def _rows(n: int = 40, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, list] = {c: [] for c in INDEX_COLUMNS}
    for i in range(n):
        data["match_id"].append(f"syn-{i % 4}")
        data["event_id"].append(i)
        data["possession_id"].append(f"p{i}")
        data["frame"].append(i * 10)
        data["team"].append("home")
        data["passer_id"].append("home_1")
        data["success"].append(int(i % 3 != 0))
        data["ball_x_att"].append(0.0)
        data["press_nearest_def_m"].append(5.0)
        data["passer_control_s"].append(1.0)
    feats = {f: rng.normal(size=n).tolist() for f in SIMILARITY_FEATURES}
    feats[SIMILARITY_FEATURES[0]] = [float(i) for i in range(n)]  # a clear 1-D ordering
    return pl.DataFrame({**data, **{k: v for k, v in feats.items() if k not in data}})


def test_query_finds_nearest_and_excludes_own_match() -> None:
    rows = _rows()
    idx = SimilarityIndex.from_rows(rows)
    x = rows.select(SIMILARITY_FEATURES).row(10)
    got, d = idx.query(np.array(x, dtype=float), exclude_match=None, k=1)
    assert got["event_id"][0] == 10 and d[0] == pytest.approx(0.0)
    got2, d2 = idx.query(np.array(x, dtype=float), exclude_match="syn-2", k=5)
    assert "syn-2" not in got2["match_id"].to_list() and (np.diff(d2) >= 0).all()


def test_missing_features_are_imputed_not_dropped() -> None:
    rows = _rows()
    idx = SimilarityIndex.from_rows(rows)
    x = np.full(len(SIMILARITY_FEATURES), np.nan)
    got, d = idx.query(x, exclude_match=None, k=3)
    assert got.height == 3 and np.isfinite(d).all()


def test_index_refuses_mixed_providers() -> None:
    with pytest.raises(ValueError, match="single provider"):
        build_index("metrica", ["metrica-sample-1", "skillcorner-1"])
