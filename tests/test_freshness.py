"""Cache freshness: a re-ingest or an index rebuild while the server runs is served next call.

Runs on a SYNTHETIC Metrica-format bundle (tests/conftest.py, not match observations)
written into a temporary data directory, so it runs in CI without the real data.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from footymind import paths
from footymind.data import store
from footymind.research import similarity
from footymind.spatial.state import causal_velocities, goalkeepers
from footymind.tracking import metrica
from footymind.tracking.model import DatasetProvenance, TrackingData
from tests.test_similarity import _rows

MID = "metrica-syn"
PROV = DatasetProvenance("metrica", "SYNTHETIC test fixture", "", "synthetic", {})


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A temporary FOOTYMIND_DATA_DIR (the data root is resolved at call time)."""
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(similarity, "_CACHE", {})
    monkeypatch.setattr(similarity, "_VALIDATED", {})
    yield tmp_path / "processed"


def _tracking(fx: dict[str, Path]) -> TrackingData:
    return metrica.load_tracking(MID, fx["home"], fx["away"], PROV)


def _write(fx: dict[str, Path], td: TrackingData | None = None) -> Path:
    # the hand-made fixture is not synchronised like real data: the sync gate is opened
    ev = metrica.load_events(fx["events"])
    return store.write_bundle(MID, td or _tracking(fx), ev, PROV, sync_gate=1e3)


def _touch(p: Path) -> None:
    """Move the file's mtime forward (coarse file-system timestamps)."""
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))


def _rewrite_manifest(root: Path, **changes: object) -> None:
    p = root / MID / "manifest.json"
    m = json.loads(p.read_text())
    m.update(changes)
    p.write_text(json.dumps(m))  # in place (same inode), as a hand edit would be
    _touch(p)


def test_rewritten_manifest_is_seen_by_the_next_call(
    data_dir: Path, synthetic_metrica: dict[str, Path]
) -> None:
    from footymind.api.app import app

    _write(synthetic_metrica)
    c = TestClient(app)
    first = c.get(f"/api/matches/{MID}").json()
    b1 = store.load_match(MID)
    assert store.load_match(MID) is b1  # cached while unchanged
    _rewrite_manifest(data_dir, counts={"events": 999, "possessions": 7})
    assert c.get(f"/api/matches/{MID}").json()["n_events"] == 999 != first["n_events"]
    b2 = store.load_match(MID)
    assert b2 is not b1 and b2.manifest["counts"]["events"] == 999


def test_pipeline_version_change_is_not_served_from_cache(
    data_dir: Path, synthetic_metrica: dict[str, Path]
) -> None:
    from footymind.api.app import app

    _write(synthetic_metrica)
    c = TestClient(app, raise_server_exceptions=False)
    assert c.get(f"/api/matches/{MID}").status_code == 200
    assert c.get(f"/api/matches/{MID}/possessions").status_code == 200
    store.load_match(MID)  # warm the bundle cache
    _rewrite_manifest(data_dir, pipeline_version="0")
    assert c.get(f"/api/matches/{MID}").status_code == 503
    r = c.get(f"/api/matches/{MID}/possessions")
    assert r.status_code == 503 and "footymind ingest" in r.json()["detail"]
    with pytest.raises(store.StaleBundleError):
        store.load_match(MID)


def test_reingest_refreshes_derived_caches(
    data_dir: Path, synthetic_metrica: dict[str, Path]
) -> None:
    """``spatial.state`` caches (goalkeepers, causal velocities) follow a re-ingest."""
    td = _tracking(synthetic_metrica)
    _write(synthetic_metrica, td)
    gk = goalkeepers(MID)
    assert gk[(1, "home")] == td.player_index("home_1")
    causal_velocities(MID)
    swapped = td.xy.copy()
    h1, h2 = td.player_index("home_1"), td.player_index("home_2")
    swapped[:, [h1, h2]] = td.xy[:, [h2, h1]]
    # a re-ingest with other positions (the manifest is replaced atomically: a new token)
    _write(synthetic_metrica, replace(td, xy=swapped))
    assert goalkeepers(MID)[(1, "home")] == h2
    assert causal_velocities(MID).shape == td.xy.shape


def _index_rows() -> pl.DataFrame:
    return _rows(seed=1).with_columns(pl.lit(MID).alias("match_id"))


def test_index_rebuild_is_picked_up(data_dir: Path, synthetic_metrica: dict[str, Path]) -> None:
    _write(synthetic_metrica)
    similarity.write_index(_index_rows(), "metrica", [MID])
    a = similarity.load_index("metrica")
    assert a is not None and similarity.load_index("metrica") is a
    meta = pl.read_parquet_metadata(similarity.index_path("metrica"))
    assert meta["footymind:pipeline_version"] == store.PIPELINE_VERSION
    assert json.loads(meta["footymind:bundles"]) == {MID: store.read_manifest(MID)["created_utc"]}
    similarity.write_index(_index_rows().head(10), "metrica", [MID])
    b = similarity.load_index("metrica")
    assert b is not None and b is not a and b.rows.height == 10


def test_index_from_another_pipeline_version_is_503(
    data_dir: Path, synthetic_metrica: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from footymind.api.app import app

    _write(synthetic_metrica)
    similarity.write_index(_index_rows(), "metrica", [MID])
    assert similarity.load_index("metrica") is not None
    # the pipeline is bumped and the bundles are re-ingested, but the index is not rebuilt
    monkeypatch.setattr(store, "PIPELINE_VERSION", "99")
    _write(synthetic_metrica)
    frame = int(store.load_match(MID).tracking.frame[0])
    r = TestClient(app).get(f"/api/matches/{MID}/frames/{frame}/similar")
    assert r.status_code == 503, r.text
    detail = r.json()["detail"]
    assert (
        "pipeline v" in detail and "footymind build-similarity-index --provider metrica" in detail
    )
    assert "metrica" not in similarity._CACHE  # the stale index is not kept


def test_index_older_than_a_reingest_is_refused(
    data_dir: Path, synthetic_metrica: dict[str, Path]
) -> None:
    _write(synthetic_metrica)
    similarity.write_index(_index_rows(), "metrica", [MID])
    assert similarity.load_index("metrica") is not None
    _rewrite_manifest(data_dir, created_utc="2099-01-01T00:00:00+00:00")
    with pytest.raises(similarity.StaleIndexError, match="re-ingest of metrica-syn"):
        similarity.load_index("metrica")


def test_index_without_build_metadata_is_refused(
    data_dir: Path, synthetic_metrica: dict[str, Path]
) -> None:
    _write(synthetic_metrica)
    p = similarity.index_path("metrica")
    _index_rows().select(similarity.STORED_COLUMNS).write_parquet(p)  # a pre-v3 index file
    with pytest.raises(similarity.StaleIndexError, match="no build metadata"):
        similarity.load_index("metrica")
