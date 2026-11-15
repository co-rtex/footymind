"""SkillCorner broadcast-tracking tests on real data (skipped if not downloaded/ingested)."""

import numpy as np
import polars as pl
import pytest

from footymind.data.store import available_matches, load_match
from footymind.spatial.state import frame_state
from footymind.tracking.model import Provenance
from tests.conftest import assert_ends_within_periods

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


def test_match_summary_has_public_match_facts_but_no_player_names() -> None:
    from fastapi.testclient import TestClient

    from footymind.api.app import app

    s = TestClient(app).get(f"/api/matches/{SC[0]}").json()
    assert s["home_label"] not in ("Home", "") and s["away_label"] not in ("Away", "")
    assert s["date"] and len(s["date"]) == 10 and s["competition"]
    import json

    from footymind.data.sources import SOURCES

    meta = json.loads(
        (SOURCES[SC[0]].local_dir() / f"{SC[0].split('-', 1)[1]}_match.json").read_text()
    )
    text = json.dumps(s)
    names = [p.get("last_name") for p in meta["players"] if p.get("last_name")]
    assert names and not any(n in text for n in names)  # no player names are served


@pytest.mark.parametrize("match_id", SC)
def test_no_event_or_possession_ends_past_its_period(match_id: str) -> None:
    """Regression (release review): SkillCorner ends crossed half-time (pipeline v3 fix)."""
    assert_ends_within_periods(load_match(match_id))
