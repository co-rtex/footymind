from pathlib import Path

import polars as pl

from footymind.tracking.metrica import load_events
from footymind.tracking.possessions import assign_possessions


def test_possession_rules(synthetic_metrica: dict[str, Path]) -> None:
    ev, pos = assign_possessions(load_events(synthetic_metrica["events"]), "syn")
    ids = ev["possession_id"].to_list()
    # kickoff set piece + pass + ball lost (home) -> p0000
    assert ids[:3] == ["syn:p0000"] * 3
    # away recovery opens p0001. The home CHALLENGE is attached to the open
    # possession and never switches it.
    assert ids[3] == "syn:p0001" and ids[4] == "syn:p0001"
    assert ids[5] == "syn:p0001" and ids[6] == "syn:p0001"  # BALL OUT closes it
    # Same team restarts with a set piece in a new period -> new possession
    assert ids[7] == ids[8] == "syn:p0002"
    assert pos.height == 3
    row = pos.filter(pl.col("possession_id") == "syn:p0001").row(0, named=True)
    assert row["team"] == "away" and row["n_events"] == 3 and row["last_type"] == "BALL OUT"


def test_possession_ids_deterministic(synthetic_metrica: dict[str, Path]) -> None:
    a = assign_possessions(load_events(synthetic_metrica["events"]), "syn")[1]
    b = assign_possessions(load_events(synthetic_metrica["events"]), "syn")[1]
    assert a.equals(b)
