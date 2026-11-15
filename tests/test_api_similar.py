"""Similar-moments endpoint (real Metrica data; skipped if absent)."""

import itertools

import pytest
from fastapi.testclient import TestClient

from footymind.data.store import available_matches
from footymind.research import similarity
from footymind.research.pass_dataset import build

pytestmark = pytest.mark.skipif(
    not {"metrica-sample-1", "metrica-sample-2"} <= set(available_matches()),
    reason="processed Metrica data not available",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    from footymind.api.app import app

    rows = build(["metrica-sample-2"], similarity.similarity_dataset_config()).select(
        similarity.STORED_COLUMNS
    )
    similarity._CACHE["metrica"] = similarity.SimilarityIndex.from_rows(rows)
    yield TestClient(app)
    similarity._CACHE.pop("metrica", None)


def test_similar_moments_contract(client: TestClient) -> None:
    q = build(["metrica-sample-1"], similarity.similarity_dataset_config()).row(30, named=True)
    body = client.get(f"/api/matches/metrica-sample-1/frames/{q['frame']}/similar?k=6").json()
    assert body["decision_frame"] == q["frame"] and body["passer_id"] == q["passer_id"]
    assert body["k"] == 6 and len(body["neighbours"]) == 6
    assert all(n["match_id"] == "metrica-sample-2" for n in body["neighbours"])  # never own match
    d = [n["distance"] for n in body["neighbours"]]
    assert d == sorted(d)
    assert body["completed_share"] == pytest.approx(
        sum(n["completed"] for n in body["neighbours"]) / 6, abs=1e-3
    )
    assert "not a prediction" in body["disclaimer"]
    est = body["estimate"]  # the validated configuration (C36): k = 50, shrunk
    assert est["k"] == 50 and 0 < est["completion"] < 1 and 0 < est["base_rate"] < 1


def test_decision_row_equals_training_row() -> None:
    q = build(["metrica-sample-1"], similarity.similarity_dataset_config()).row(12, named=True)
    row = similarity.decision_row("metrica-sample-1", int(q["frame"]))
    assert row is not None and row["event_id"] == q["event_id"]
    for f in similarity.SIMILARITY_FEATURES:
        assert row[f] == pytest.approx(q[f], nan_ok=True), f


def test_no_decision_nearby_is_404(client: TestClient) -> None:
    from footymind.data.store import load_match

    starts = sorted(load_match("metrica-sample-1").events["start_frame"].unique().to_list())
    gap = next(a for a, b in itertools.pairwise(starts) if b - a > 100)  # > 4 s
    r = client.get(f"/api/matches/metrica-sample-1/frames/{gap + 1}/similar")
    assert r.status_code == 404
