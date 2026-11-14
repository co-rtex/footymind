"""Canonical in-memory representation of tracked football matches.

Coordinate convention (documented in docs/ARCHITECTURE.md#coordinates)
------------------------------------------------------------------------
* Units: metres.
* Origin: centre spot. ``x`` runs along the pitch length in
  ``[-L/2, L/2]``; ``y`` runs along the width in ``[-W/2, W/2]`` with **y
  pointing "up"** when the pitch is drawn with x to the right (a standard
  right-handed frame).
* Orientation is the *provider's* orientation: teams swap ends at half-time
  and the coordinates are **not** flipped. ``TrackingData.home_attacks_positive_x``
  records, per period, whether the home team attacks towards +x. Use
  :func:`footymind.tracking.coords.attacking_frame_sign` to rotate into an
  "attacking left-to-right" frame for analytics.

Observation semantics
---------------------
Unobserved positions are ``NaN``. They are never filled with fabricated
values. Any interpolated or estimated quantity is produced by an explicit
function and is labelled as derived or estimated by the caller (see
:class:`Provenance`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

import numpy as np
import numpy.typing as npt

TeamSide = Literal["home", "away"]

FloatArray = npt.NDArray[np.float32]
F64Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class Provenance(StrEnum):
    """Epistemic status of a quantity. Carried through the API and the UI."""

    OBSERVED = "observed"  # provider measurement, transformed only by fixed unit conversion
    DERIVED = "derived"  # deterministic function of observations (e.g. finite-difference velocity)
    ESTIMATED = "estimated"  # output of a model with assumptions/parameters (e.g. pitch control)
    HYPOTHETICAL = "hypothetical"  # user/system intervention applied to an observed state
    SIMULATED = "simulated"  # trajectories or responses produced by a simulation policy


@dataclass(frozen=True)
class PitchSpec:
    length: float = 105.0
    width: float = 68.0

    @property
    def half_length(self) -> float:
        return self.length / 2.0

    @property
    def half_width(self) -> float:
        return self.width / 2.0


@dataclass(frozen=True)
class PlayerInfo:
    """Static identity of a tracked player.

    ``player_id`` is stable within the dataset (``"home_11"``). ``provider_id``
    is the provider's own label (``"Player11"``).
    """

    player_id: str
    team: TeamSide
    jersey: int
    provider_id: str


@dataclass(frozen=True)
class DatasetProvenance:
    provider: str
    source_name: str
    source_url: str
    license_note: str
    files: dict[str, str]  # filename -> sha256


@dataclass
class TrackingData:
    """Dense tracking arrays for one match.

    Shapes: ``T`` frames, ``P`` players.
    """

    match_id: str
    pitch: PitchSpec
    fps: float
    frame: IntArray  # (T,) provider frame ids, strictly increasing
    period: npt.NDArray[np.int8]  # (T,)
    time_s: F64Array  # (T,) provider match clock in seconds
    players: list[PlayerInfo]
    xy: FloatArray  # (T, P, 2) metres, NaN when not observed
    ball_xy: FloatArray  # (T, 2) metres, NaN when not observed
    home_attacks_positive_x: dict[int, bool]
    provenance: DatasetProvenance | None = None
    # (T, P) True where the provider *detected* the player. Broadcast tracking (SkillCorner)
    # also reports extrapolated positions for players outside the camera view; those are
    # estimates, not observations. None means all non-NaN positions are observed (Metrica).
    detected: npt.NDArray[np.bool_] | None = None
    # (T,) True where the provider detected the ball. None: every non-NaN ball position is
    # observed (Metrica, or a bundle ingested before this field existed).
    ball_detected: npt.NDArray[np.bool_] | None = None
    _frame_index: dict[int, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        t = self.frame.shape[0]
        p = len(self.players)
        if self.xy.shape != (t, p, 2):
            raise ValueError(f"xy shape {self.xy.shape} != {(t, p, 2)}")
        if self.ball_xy.shape != (t, 2):
            raise ValueError(f"ball_xy shape {self.ball_xy.shape} != {(t, 2)}")
        if self.period.shape != (t,) or self.time_s.shape != (t,):
            raise ValueError("period/time_s must have shape (T,)")
        if t > 1 and not np.all(np.diff(self.frame) > 0):
            raise ValueError("frame ids must be strictly increasing")
        ids = [pl.player_id for pl in self.players]
        if len(set(ids)) != len(ids):
            raise ValueError("player_id values must be unique")

    @property
    def n_frames(self) -> int:
        return int(self.frame.shape[0])

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    def index_of_frame(self, frame_id: int) -> int:
        """Row index for a provider frame id (KeyError if absent)."""
        if not self._frame_index:
            self._frame_index = {int(f): i for i, f in enumerate(self.frame)}
        return self._frame_index[int(frame_id)]

    def team_indices(self, team: TeamSide) -> IntArray:
        return np.array([i for i, p in enumerate(self.players) if p.team == team], dtype=np.int64)

    def player_index(self, player_id: str) -> int:
        for i, p in enumerate(self.players):
            if p.player_id == player_id:
                return i
        raise KeyError(player_id)

    def attack_sign(self, team: TeamSide, period: int) -> int:
        """+1 if ``team`` attacks towards +x in ``period``, else -1."""
        home_pos = self.home_attacks_positive_x[int(period)]
        sign = 1 if home_pos else -1
        return sign if team == "home" else -sign
