"""Shared test fixtures.

Synthetic fixtures here are hand-constructed in the Metrica CSV format. They are
NOT match observations. They exist to test parsing and edge cases
deterministically.
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path

import pytest

from footymind.paths import PROCESSED_DIR, RAW_DIR

METRICA_RAW = RAW_DIR / "metrica"


def has_metrica(game: int = 1) -> bool:
    return (METRICA_RAW / f"Sample_Game_{game}_RawEventsData.csv").exists()


requires_metrica = pytest.mark.skipif(
    not has_metrica(1) or not has_metrica(2),
    reason="Metrica sample data not downloaded (run `footymind download`)",
)


def _tracking_csv(team: str, players: list[tuple[str, int]], rows: list[list[str]]) -> str:
    h1 = ["", "", "", *chain.from_iterable([team, ""] for _ in players), "", ""]
    h2 = ["", "", "", *chain.from_iterable([str(j), ""] for _, j in players), "", ""]
    h3 = ["Period", "Frame", "Time [s]", *chain.from_iterable([n, ""] for n, _ in players)]
    h3 += ["Ball", ""]
    return "\n".join(",".join(r) for r in [h1, h2, h3, *rows]) + "\n"


@pytest.fixture
def synthetic_metrica(tmp_path: Path) -> dict[str, Path]:
    """Two tiny synthetic Metrica-format tracking files and one event file.

    Home: GK "Player1" (left half in period 1) and "Player 2" (note the space,
    mirroring a real Game 2 quirk). Away: GK "Player3" (right half) and
    "Player4", whose position is NaN in frames 1-2 (not yet on pitch).
    Two periods; teams switch ends in period 2.
    """
    home_rows, away_rows = [], []
    for f in range(1, 9):
        per = 1 if f <= 4 else 2
        t = f * 0.04
        flip = per == 2

        def X(v: float, flip: bool = flip) -> str:
            return f"{(1 - v) if flip else v:.5f}"

        ball = [X(0.5 + 0.01 * f), "0.50000"]
        home_rows.append(
            [str(per), str(f), f"{t:.2f}", X(0.02), "0.50000", X(0.40), "0.25000", *ball]
        )
        p4 = ["NaN", "NaN"] if f <= 2 else [X(0.60), "0.75000"]
        away_rows.append([str(per), str(f), f"{t:.2f}", X(0.98), "0.50000", *p4, *ball])
    home = tmp_path / "home.csv"
    away = tmp_path / "away.csv"
    home.write_text(_tracking_csv("Home", [("Player1", 1), ("Player 2", 2)], home_rows))
    away.write_text(_tracking_csv("Away", [("Player3", 3), ("Player4", 4)], away_rows))
    events = tmp_path / "events.csv"
    events.write_text(
        "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,"
        "Start X,Start Y,End X,End Y\n"
        "Home,SET PIECE,KICK OFF,1,1,0.04,0,0,Player 2,,NaN,NaN,NaN,NaN\n"
        "Home,PASS,,1,1,0.04,3,0.12,Player 2,Player1,0.40,0.25,0.02,0.50\n"
        "Home,BALL LOST,INTERCEPTION,1,3,0.12,4,0.16,Player1,,0.02,0.50,0.6,0.5\n"
        "Away,RECOVERY,INTERCEPTION,1,4,0.16,0,0,Player4,,0.6,0.5,NaN,NaN\n"
        "Home,CHALLENGE,GROUND-LOST,1,4,0.16,0,0,Player1,,0.6,0.5,NaN,NaN\n"
        "Away,PASS,,1,4,0.16,4,0.16,Player4,Player3,0.6,0.75,0.98,0.5\n"
        "Away,BALL OUT,,1,4,0.16,4,0.16,Player3,,0.98,0.5,1.02,0.5\n"
        "Away,SET PIECE,THROW IN,2,5,0.20,0,0,Player3,,NaN,NaN,NaN,NaN\n"
        "Away,PASS,,2,5,0.20,6,0.24,Player3,Player4,0.02,0.5,0.4,0.25\n"
    )
    return {"home": home, "away": away, "events": events}


@pytest.fixture(scope="session")
def processed_root() -> Path:
    return PROCESSED_DIR
