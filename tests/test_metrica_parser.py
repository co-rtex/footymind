from pathlib import Path

import numpy as np
import polars as pl
import pytest

from footymind.tracking.metrica import load_events, load_tracking


def test_tracking_parse(synthetic_metrica: dict[str, Path]) -> None:
    td = load_tracking("syn", synthetic_metrica["home"], synthetic_metrica["away"])
    assert td.xy.shape == (8, 4, 2)
    assert [p.player_id for p in td.players] == ["home_1", "home_2", "away_3", "away_4"]
    assert td.players[1].provider_id == "Player 2"
    # Home GK at x=0.02 in period 1 -> (0.02-0.5)*105
    assert td.xy[0, 0, 0] == pytest.approx(-50.4, abs=1e-3)
    # Substitute not yet on the pitch stays NaN (never fabricated)
    assert np.isnan(td.xy[0, 3]).all() and np.isfinite(td.xy[2, 3]).all()
    # Home GK on the -x side in period 1 -> home attacks +x; ends switch in period 2
    assert td.home_attacks_positive_x == {1: True, 2: False}
    assert td.attack_sign("away", 1) == -1


def test_tracking_rejects_misaligned_files(
    synthetic_metrica: dict[str, Path], tmp_path: Path
) -> None:
    lines = synthetic_metrica["away"].read_text().splitlines()
    bad = tmp_path / "away_bad.csv"
    bad.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="frame-aligned"):
        load_tracking("syn", synthetic_metrica["home"], bad)


def test_events_parse(synthetic_metrica: dict[str, Path]) -> None:
    ev = load_events(synthetic_metrica["events"])
    assert ev.height == 9
    # "Player 2" with a space resolves to the canonical id (Game 2 regression)
    assert ev["from_player_id"][1] == "home_2"
    assert ev["to_player_id"][1] == "home_1"
    # End Frame 0 means "no end" -> null
    assert ev["end_frame"][0] is None
    kick = ev.row(1, named=True)
    assert kick["start_x"] == pytest.approx((0.40 - 0.5) * 105)
    assert kick["start_y"] == pytest.approx(-(0.25 - 0.5) * 68)
    assert ev.filter(pl.col("type") == "SET PIECE")["end_time_s"].null_count() == 2
