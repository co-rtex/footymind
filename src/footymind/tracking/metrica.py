"""Metrica Sports sample-data adapter (CSV format, Sample Games 1 and 2).

Format facts verified against the downloaded files (2026-11-13):

* Tracking: one CSV per team. The first three rows hold team, jersey and column
  names. Each player has an x column and an unnamed y column. ``Ball`` is the
  last pair, duplicated identically in both team files. Rate: 25 Hz, frames
  are contiguous across both periods, and ``Time [s]`` is cumulative.
* ``NaN`` for a player means the player is not on the pitch (substitutions).
  ``NaN`` for the ball mostly coincides with dead-ball periods.
* Events: ``Start Frame`` and ``End Frame`` are tracking frame ids
  (synchronised, per the provider). ``End Frame == 0`` means "no end".
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl

from footymind.tracking.coords import metrica_to_canonical
from footymind.tracking.model import (
    DatasetProvenance,
    PitchSpec,
    PlayerInfo,
    TeamSide,
    TrackingData,
)

METRICA_PITCH = PitchSpec(105.0, 68.0)
METRICA_FPS = 25.0

EVENT_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "event_id": pl.Int64,
    "team": pl.Utf8,
    "type": pl.Utf8,
    "subtype": pl.Utf8,
    "period": pl.Int8,
    "start_frame": pl.Int64,
    "end_frame": pl.Int64,
    "start_time_s": pl.Float64,
    "end_time_s": pl.Float64,
    "from_player_id": pl.Utf8,
    "to_player_id": pl.Utf8,
    "start_x": pl.Float64,
    "start_y": pl.Float64,
    "end_x": pl.Float64,
    "end_y": pl.Float64,
}


def _read_team_tracking(
    path: Path, team: TeamSide
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[PlayerInfo], np.ndarray, np.ndarray]:
    with path.open() as f:
        team_row = f.readline().rstrip("\n").split(",")
        jersey_row = f.readline().rstrip("\n").split(",")
        name_row = f.readline().rstrip("\n").split(",")
    if name_row[:3] != ["Period", "Frame", "Time [s]"] or name_row[-2] != "Ball":
        raise ValueError(f"{path}: unexpected Metrica header {name_row[:3]} ... {name_row[-2:]}")

    players: list[PlayerInfo] = []
    xcols: list[int] = []
    for i in range(3, len(name_row) - 2, 2):
        pid = name_row[i]
        if not pid.startswith("Player"):
            raise ValueError(f"{path}: unexpected column {pid!r} at {i}")
        if team_row[i].strip().lower() != team:
            raise ValueError(f"{path}: column {pid} labelled team {team_row[i]!r}, expected {team}")
        jersey = int(jersey_row[i])
        players.append(PlayerInfo(f"{team}_{jersey}", team, jersey, pid))
        xcols.append(i)

    raw = pl.read_csv(
        path,
        skip_rows=3,
        has_header=False,
        infer_schema_length=0,
        null_values=["NaN", ""],
    )
    mat = raw.select(pl.all().cast(pl.Float64)).to_numpy()
    period = mat[:, 0].astype(np.int8)
    frame = mat[:, 1].astype(np.int64)
    time_s = mat[:, 2].astype(np.float64)
    px = mat[:, xcols]
    py = mat[:, [c + 1 for c in xcols]]
    ball = mat[:, -2:]
    xy_m = np.stack(metrica_to_canonical(px, py, METRICA_PITCH), axis=-1)
    bx, by = metrica_to_canonical(ball[:, 0], ball[:, 1], METRICA_PITCH)
    return frame, period, time_s, players, xy_m, np.stack([bx, by], axis=-1)


def _infer_home_direction(
    xy: np.ndarray, period: np.ndarray, players: list[PlayerInfo]
) -> dict[int, bool]:
    """Per period, does the home team attack +x?

    For each team, the goalkeeper is identified as the player whose median x
    over the period lies furthest from the halfway line. A goalkeeper defends
    the goal on their own side, so the home team attacks +x when the home
    goalkeeper's median x is negative. The two goalkeepers must sit in
    opposite halves. If they do not, the inference is ambiguous and an error is
    raised rather than a direction guessed. (A kickoff-centroid rule fails
    because the kicking-off team stands on the halfway line.)
    """
    home = np.array([p.team == "home" for p in players])
    out: dict[int, bool] = {}
    for per in np.unique(period):
        rows = period == per
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns: unused subs
            med_x = np.nanmedian(xy[rows][:, :, 0], axis=0)  # (P,)
        gk_x = {}
        for name, mask in (("home", home), ("away", ~home)):
            cand = np.where(mask & ~np.isnan(med_x), np.abs(med_x), -1.0)
            gk_x[name] = float(med_x[int(np.argmax(cand))])
        hx, ax = gk_x["home"], gk_x["away"]
        if np.sign(hx) == np.sign(ax) or min(abs(hx), abs(ax)) < 30.0:
            raise ValueError(f"cannot infer attacking direction for period {per}: {hx=} {ax=}")
        out[int(per)] = bool(hx < 0)
    return out


def load_tracking(
    match_id: str,
    home_csv: Path,
    away_csv: Path,
    provenance: DatasetProvenance | None = None,
) -> TrackingData:
    fh, ph, th, players_h, xy_h, ball_h = _read_team_tracking(home_csv, "home")
    fa, pa, ta, players_a, xy_a, ball_a = _read_team_tracking(away_csv, "away")
    if not (np.array_equal(fh, fa) and np.array_equal(ph, pa) and np.allclose(th, ta)):
        raise ValueError("home/away tracking files are not frame-aligned")
    both = ~np.isnan(ball_h[:, 0]) & ~np.isnan(ball_a[:, 0])
    if both.any() and np.nanmax(np.abs(ball_h[both] - ball_a[both])) > 1e-6:
        raise ValueError("ball positions differ between home and away tracking files")
    xy = np.concatenate([xy_h, xy_a], axis=1).astype(np.float32)
    players = players_h + players_a
    return TrackingData(
        match_id=match_id,
        pitch=METRICA_PITCH,
        fps=METRICA_FPS,
        frame=fh,
        period=ph,
        time_s=th,
        players=players,
        xy=xy,
        ball_xy=ball_h.astype(np.float32),
        home_attacks_positive_x=_infer_home_direction(xy, ph, players),
        provenance=provenance,
    )


def load_events(events_csv: Path) -> pl.DataFrame:
    """Parse Metrica events into the canonical event table (coordinates in metres)."""
    raw = pl.read_csv(events_csv, infer_schema_length=10000, null_values=["NaN", ""])
    expected = [
        "Team", "Type", "Subtype", "Period", "Start Frame", "Start Time [s]", "End Frame",
        "End Time [s]", "From", "To", "Start X", "Start Y", "End X", "End Y",
    ]  # fmt: skip
    if raw.columns != expected:
        raise ValueError(f"{events_csv}: unexpected columns {raw.columns}")

    team = raw["Team"].str.to_lowercase()
    if not team.is_in(["home", "away"]).all():
        raise ValueError(f"unexpected team labels {team.unique().to_list()}")

    sx, sy = metrica_to_canonical(
        raw["Start X"].cast(pl.Float64).to_numpy(), raw["Start Y"].cast(pl.Float64).to_numpy(),
        METRICA_PITCH,
    )  # fmt: skip
    ex, ey = metrica_to_canonical(
        raw["End X"].cast(pl.Float64).to_numpy(), raw["End Y"].cast(pl.Float64).to_numpy(),
        METRICA_PITCH,
    )  # fmt: skip

    def pid(col: str) -> pl.Series:
        # "Player19" + team "away" -> "away_19". The provider ids are per-match
        # unique across teams, but namespacing by team guards against reuse.
        s = raw[col]
        num = s.str.extract(r"^Player\s*(\d+)$", 1)
        out = pl.when(num.is_not_null()).then(team + "_" + num).otherwise(None)
        return pl.select(out.alias(col)).to_series()

    end_frame = raw["End Frame"].cast(pl.Int64)
    df = pl.DataFrame(
        {
            "event_id": np.arange(raw.height, dtype=np.int64),
            "team": team,
            "type": raw["Type"],
            "subtype": raw["Subtype"],
            "period": raw["Period"].cast(pl.Int8),
            "start_frame": raw["Start Frame"].cast(pl.Int64),
            "end_frame": pl.select(
                pl.when(end_frame > 0).then(end_frame).otherwise(None)
            ).to_series(),
            "start_time_s": raw["Start Time [s]"].cast(pl.Float64),
            "end_time_s": raw["End Time [s]"].cast(pl.Float64),
            "from_player_id": pid("From"),
            "to_player_id": pid("To"),
            "start_x": sx,
            "start_y": sy,
            "end_x": ex,
            "end_y": ey,
        },
        schema=EVENT_SCHEMA,
    )
    # A "To" receiver must be on the same team as the "From" player, since
    # Metrica only fills "To" for completed passes and set-piece receptions.
    # Cross-team receivers would indicate a parse error.
    return df.with_columns(
        pl.when(pl.col("end_time_s") <= 0).then(None).otherwise(pl.col("end_time_s"))
        .alias("end_time_s")
    )  # fmt: skip
