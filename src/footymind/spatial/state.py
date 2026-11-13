"""Extraction of an analysable game state at a single frame."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
import numpy.typing as npt
import polars as pl

from footymind.data.store import MatchBundle, load_match
from footymind.tracking.kinematics import clean_velocity
from footymind.tracking.model import Provenance, TeamSide, TrackingData

F = npt.NDArray[np.float64]

CARRIER_RADIUS_M = 1.5  # ball within this distance of a player -> that player carries it


@lru_cache(maxsize=4)
def causal_velocities(match_id: str) -> npt.NDArray[np.float32]:
    """Causal (backward 0.2 s) velocities for every frame, cleaned of artefacts."""
    td = load_match(match_id).tracking
    return clean_velocity(td.xy, td.fps, "causal", 0.2, td.period)


@lru_cache(maxsize=4)
def goalkeepers(match_id: str) -> dict[tuple[int, TeamSide], int]:
    """Player index of each team's goalkeeper per period (deepest median position)."""
    td = load_match(match_id).tracking
    out: dict[tuple[int, TeamSide], int] = {}
    for per in np.unique(td.period):
        rows = td.period == per
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(td.xy[rows][:, :, 0], axis=0)
        for team in ("home", "away"):
            idx = td.team_indices(team)
            sign = td.attack_sign(team, int(per))
            # The GK sits deepest towards their own goal: the minimum of x * sign
            vals = np.where(np.isnan(med[idx]), np.inf, med[idx] * sign)
            out[(int(per), team)] = int(idx[int(np.argmin(vals))])
    return out


@dataclass
class FrameState:
    match_id: str
    frame: int
    period: int
    time_s: float
    player_ids: list[str]
    teams: list[TeamSide]
    jerseys: list[int]
    xy: F  # (P, 2), NaN = not on pitch / unobserved
    vel: F  # (P, 2), NaN = unavailable
    is_gk: npt.NDArray[np.bool_]
    ball: F  # (2,), NaN if unobserved
    attacking_team: TeamSide | None
    possession_id: str | None
    carrier: int | None
    attack_sign: int  # +1 if attacking team attacks +x in this period
    position_provenance: list[Provenance] = field(default_factory=list)
    velocity_provenance: list[Provenance] = field(default_factory=list)

    def copy(self) -> FrameState:
        return replace(
            self,
            xy=self.xy.copy(),
            vel=self.vel.copy(),
            position_provenance=list(self.position_provenance),
            velocity_provenance=list(self.velocity_provenance),
        )

    def index(self, player_id: str) -> int:
        return self.player_ids.index(player_id)

    def team_mask(self, team: TeamSide) -> npt.NDArray[np.bool_]:
        return np.array([t == team for t in self.teams])

    def on_pitch(self) -> npt.NDArray[np.bool_]:
        return ~np.isnan(self.xy[:, 0])


def attacking_team_at(bundle: MatchBundle, frame: int) -> tuple[TeamSide | None, str | None]:
    pos = bundle.possessions
    hit = pos.filter((pl.col("start_frame") <= frame) & (pl.col("end_frame") >= frame))
    if hit.height == 0:
        hit = pos.filter(pl.col("start_frame") <= frame).tail(1)
    if hit.height == 0:
        return None, None
    row = hit.tail(1).row(0, named=True)
    return row["team"], row["possession_id"]


def find_carrier(xy: F, ball: F, teams: list[TeamSide], team: TeamSide | None) -> int | None:
    if team is None or np.isnan(ball).any():
        return None
    d = np.linalg.norm(xy - ball[None, :], axis=1)
    mask = np.array([t == team for t in teams]) & ~np.isnan(d)
    if not mask.any():
        return None
    i = int(np.argmin(np.where(mask, d, np.inf)))
    return i if d[i] <= CARRIER_RADIUS_M else None


ON_BALL_TYPES = ("PASS", "SHOT", "BALL LOST", "SET PIECE")
EVENT_CARRIER_TOLERANCE_FRAMES = 3


def event_carrier(bundle: MatchBundle, frame: int, team: TeamSide) -> int | None:
    """Fallback: the player who starts an on-ball event within a few frames of ``frame``.

    Event start annotations occasionally lag the kick (the ball has already
    travelled a few metres), so the proximity rule alone misses the passer.
    """
    ev = bundle.events.filter(
        (pl.col("team") == team)
        & pl.col("type").is_in(ON_BALL_TYPES)
        & ((pl.col("start_frame") - frame).abs() <= EVENT_CARRIER_TOLERANCE_FRAMES)
        & pl.col("from_player_id").is_not_null()
    )
    if ev.height == 0:
        return None
    pid = ev.sort((pl.col("start_frame") - frame).abs()).row(0, named=True)["from_player_id"]
    return bundle.tracking.player_index(pid)


def frame_state(match_id: str, frame: int) -> FrameState:
    bundle = load_match(match_id)
    td: TrackingData = bundle.tracking
    i = td.index_of_frame(frame)
    period = int(td.period[i])
    xy = td.xy[i].astype(np.float64)
    vel = causal_velocities(match_id)[i].astype(np.float64)
    gk = goalkeepers(match_id)
    is_gk = np.zeros(len(td.players), dtype=bool)
    is_gk[gk[(period, "home")]] = True
    is_gk[gk[(period, "away")]] = True
    team, pid = attacking_team_at(bundle, frame)
    teams: list[TeamSide] = [p.team for p in td.players]
    ball = td.ball_xy[i].astype(np.float64)
    carrier = find_carrier(xy, ball, teams, team)
    if carrier is None and team is not None:
        carrier = event_carrier(bundle, frame, team)
    # Velocity is always a derived quantity. A null value means it could not be derived
    # (player absent, window incomplete, or implausible-motion artefact).
    vprov = [Provenance.DERIVED] * len(td.players)
    return FrameState(
        match_id=match_id,
        frame=int(frame),
        period=period,
        time_s=float(td.time_s[i]),
        player_ids=[p.player_id for p in td.players],
        teams=teams,
        jerseys=[p.jersey for p in td.players],
        xy=xy,
        vel=vel,
        is_gk=is_gk,
        ball=ball,
        attacking_team=team,
        possession_id=pid,
        carrier=carrier,
        attack_sign=td.attack_sign(team, period) if team else 1,
        position_provenance=[Provenance.OBSERVED] * len(td.players),
        velocity_provenance=vprov,
    )
