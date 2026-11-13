"""SkillCorner broadcast-tracking tests on real data (skipped if not downloaded/ingested)."""

import numpy as np
import polars as pl
import pytest

from footymind.data.store import available_matches, load_match
from footymind.spatial.state import frame_state
from footymind.tracking.model import Provenance

SC = [m for m in available_matches() if m.startswith("skillcorner-")]
pytestmark = pytest.mark.skipif(not SC, reason="SkillCorner data not ingested")


@pytest.fixture(scope="module")
def bundle():  # type: ignore[no-untyped-def]
    return load_match(SC[0])


def test_detected_mask_and_provenance(bundle) -> None:  # type: ignore[no-untyped-def]
    td = bundle.tracking
    assert td.detected is not None and td.detected.shape == td.xy.shape[:2]
    # a detected flag never marks an absent position
    assert not np.any(td.detected & np.isnan(td.xy[..., 0]))
    # extrapolated positions are reported as estimated, never observed
    passes = bundle.events.filter(pl.col("type") == "PASS")
    f = int(passes["start_frame"][len(passes) // 2])
    st = frame_state(SC[0], f)
    i = td.index_of_frame(f)
    for k, prov in enumerate(st.position_provenance):
        if np.isnan(td.xy[i, k, 0]):
            continue
        assert prov == (Provenance.OBSERVED if td.detected[i, k] else Provenance.ESTIMATED)


def test_sync_and_direction(bundle) -> None:  # type: ignore[no-untyped-def]
    q = bundle.manifest["quality"]["event_sync"]["passer_to_ball_at_start_m"]
    assert q["median"] < 2.5 and q["n"] > 300
    assert set(bundle.tracking.home_attacks_positive_x) == {1, 2}
    assert bundle.tracking.home_attacks_positive_x[1] != bundle.tracking.home_attacks_positive_x[2]
    assert bundle.manifest["provenance"]["provider"] == "skillcorner"


def test_replay_exposes_detected_mask() -> None:
    from fastapi.testclient import TestClient

    from footymind.api.app import app

    c = TestClient(app)
    pos = c.get(f"/api/matches/{SC[0]}/possessions", params={"min_passes": 3}).json()
    r = c.get(f"/api/possessions/{pos[0]['possession_id']}/replay").json()
    n = len(r["frame"]) * len(r["players"])
    assert r["players_detected"] is not None and len(r["players_detected"]) == n
    assert not all(r["players_detected"])  # broadcast tracking: some positions extrapolated
