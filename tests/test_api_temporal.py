"""Served temporal context and explanations (real Metrica data; skipped if absent)."""

import pytest
from fastapi.testclient import TestClient

from footymind.data.store import available_matches
from footymind.research.pass_dataset import build

pytestmark = pytest.mark.skipif(
    "metrica-sample-1" not in available_matches(), reason="processed Metrica data not available"
)
MID = "metrica-sample-1"


@pytest.fixture(scope="module")
def client() -> TestClient:
    from footymind.api.app import app

    return TestClient(app)


def test_served_temporal_matches_training_features(client: TestClient) -> None:
    """Train/serve consistency: the API's temporal block equals the research dataset's columns
    whenever the API's carrier is the event's passer."""
    df = build([MID]).head(40)
    checked = 0
    for r in df.iter_rows(named=True):
        body = client.get(f"/api/matches/{MID}/frames/{r['frame']}/analysis?grid=false").json()
        if body["carrier_id"] != r["passer_id"] or body["temporal"] is None:
            continue
        for k, v in body["temporal"].items():
            if v is None:
                assert r[k] != r[k], k  # NaN in training is null when served
            else:
                assert v == pytest.approx(r[k], abs=1e-3), k
        checked += 1
    assert checked >= 20  # carrier detection agrees with the event passer for most passes


def test_explanations_contract(client: TestClient) -> None:
    df = build([MID]).head(5)
    body = client.get(f"/api/matches/{MID}/frames/{df['frame'][0]}/analysis?grid=false").json()
    ex = body["explanations"]
    assert ex and {e["kind"] for e in ex} <= {"pressure", "trend", "option", "caveat"}
    ids = {p["player_id"] for p in body["players"]}
    for e in ex:
        assert e["metric"] and e["provenance"] in {"observed", "derived", "estimated"}
        assert set(e["players"]) <= ids
