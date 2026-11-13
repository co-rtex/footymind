"""Tests against the real Metrica sample data (skipped if not downloaded)."""

import numpy as np
import polars as pl
import pytest

from footymind.data import sources
from footymind.data.store import ingest, load_match
from tests.conftest import METRICA_RAW, requires_metrica

pytestmark = [requires_metrica, pytest.mark.data]


GAMES = ["metrica-sample-1", "metrica-sample-2"] + (
    ["metrica-sample-3"] if (METRICA_RAW / "Sample_Game_3_tracking.txt").exists() else []
)


@pytest.fixture(scope="module", params=GAMES)
def bundle(request, tmp_path_factory):  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("processed")
    ingest(request.param, out_root=out)
    load_match.cache_clear()
    return load_match(request.param, root=out)


def test_checksums_pinned() -> None:
    sources.verify(sources.SOURCES["metrica-sample-1"])
    sources.verify(sources.SOURCES["metrica-sample-2"])


def test_known_source_frame_transform() -> None:
    """Game 1 frame 1, home Player11 raw (0.00082, 0.48238) from the provider CSV."""
    load_match.cache_clear()
    from footymind.tracking.metrica import load_tracking

    td = load_tracking(
        "g1",
        METRICA_RAW / "Sample_Game_1_RawTrackingData_Home_Team.csv",
        METRICA_RAW / "Sample_Game_1_RawTrackingData_Away_Team.csv",
    )
    i = td.player_index("home_11")
    np.testing.assert_allclose(
        td.xy[0, i], [(0.00082 - 0.5) * 105, -(0.48238 - 0.5) * 68], atol=1e-4
    )
    assert td.fps == 25.0 and np.all(np.diff(td.frame) == 1)


def test_event_tracking_synchronisation(bundle) -> None:  # type: ignore[no-untyped-def]
    sync = bundle.manifest["quality"]["event_sync"]
    assert sync["passer_to_ball_at_start_m"]["median"] < 0.25
    assert sync["receiver_to_ball_at_end_m"]["median"] < 0.5
    assert sync["passer_to_ball_at_start_m"]["n"] > 700


def test_goals_scored_towards_attacking_goal(bundle) -> None:  # type: ignore[no-untyped-def]
    td = bundle.tracking
    goals = bundle.events.filter(
        (pl.col("type") == "SHOT") & pl.col("subtype").fill_null("").str.contains("GOAL")
    )
    assert goals.height >= 1
    for r in goals.iter_rows(named=True):
        assert r["start_x"] * td.attack_sign(r["team"], r["period"]) > 30.0


def test_possessions_partition_events(bundle) -> None:  # type: ignore[no-untyped-def]
    ev, pos = bundle.events, bundle.possessions
    assert pos["possession_id"].is_unique().all()
    assert ev.filter(pl.col("type") == "PASS")["possession_id"].null_count() == 0
    assert (pos["end_frame"] >= pos["start_frame"]).all()
