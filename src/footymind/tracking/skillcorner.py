"""SkillCorner open-data adapter (broadcast tracking, A-League 2024/25).

Format facts verified against the downloaded files (2026-11-13):

* ``{id}_match.json``: players (``id``, ``team_id``, ``number``, ``player_role``),
  ``home_team_side`` per period (``left_to_right`` = attacking +x), ``match_periods`` with
  frame ranges, and pitch length/width.
* ``{id}_tracking_extrapolated.jsonl``: one JSON object per video frame at **10 fps**.
  Coordinates are metres with the origin at the centre (x along the length). Each
  ``player_data`` entry has ``player_id`` (= ``players[].id``) and ``is_detected``.
  Positions with ``is_detected = false`` are the provider's **extrapolations** for players
  outside the camera view, so they are estimates. Many in-period frames have no
  ``player_data`` at all (broadcast replays or close-ups). Those positions stay NaN.
* ``{id}_dynamic_events.csv``: ``player_possession`` rows with ``end_type`` and
  ``pass_outcome``. ``frame_end`` is when the possession ended, i.e. the release for a pass.

Orientation: the attacking direction declared in ``home_team_side`` is cross-checked
against goalkeeper positions, and a contradiction raises. The sign convention of the
SkillCorner y axis could not be verified against video. Our features use ``|y|`` or
rotation-invariant quantities, so a y-mirror would not change them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from footymind.tracking.metrica import EVENT_SCHEMA, _infer_home_direction
from footymind.tracking.model import (
    DatasetProvenance,
    PitchSpec,
    PlayerInfo,
    TeamSide,
    TrackingData,
)

SKILLCORNER_FPS = 10.0


def _players(meta: dict[str, Any]) -> tuple[list[PlayerInfo], dict[int, int], dict[int, TeamSide]]:
    home_id = meta["home_team"]["id"]
    side: dict[int, TeamSide] = {
        home_id: "home",
        meta["away_team"]["id"]: "away",
    }
    players: list[PlayerInfo] = []
    index: dict[int, int] = {}
    seen: set[str] = set()
    for p in sorted(meta["players"], key=lambda p: (p["team_id"] != home_id, p["number"], p["id"])):
        team = side[p["team_id"]]
        pid = f"{team}_{p['number']}"
        if pid in seen:  # duplicate shirt numbers are disambiguated, never merged
            pid = f"{pid}_{p['id']}"
        seen.add(pid)
        index[int(p["id"])] = len(players)
        # provider_id is the numeric SkillCorner id; names are deliberately not stored
        players.append(PlayerInfo(pid, team, int(p["number"]), str(p["id"])))
    return players, index, side


def load_tracking(
    match_id: str, tracking_jsonl: Path, match_json: Path, provenance: DatasetProvenance | None
) -> tuple[TrackingData, dict[str, Any]]:
    meta = json.loads(match_json.read_text())
    players, index, _ = _players(meta)
    periods = [
        (int(p["period"]), int(p["start_frame"]), int(p["end_frame"]))
        for p in meta["match_periods"]
    ]
    rows: list[tuple[int, int]] = []
    pos: list[np.ndarray] = []
    det: list[np.ndarray] = []
    balls: list[tuple[float, float]] = []
    p_n = len(players)
    with tracking_jsonl.open() as fh:
        for line in fh:
            r = json.loads(line)
            f = int(r["frame"])
            per = next((k for k, a, b in periods if a <= f <= b), None)
            if per is None:
                continue
            xy = np.full((p_n, 2), np.nan, dtype=np.float32)
            d = np.zeros(p_n, dtype=bool)
            for pd in r.get("player_data") or []:
                i = index.get(int(pd["player_id"]))
                if i is None or pd["x"] is None or pd["y"] is None:
                    continue
                xy[i] = (pd["x"], pd["y"])
                d[i] = bool(pd.get("is_detected"))
            b = r.get("ball_data") or {}
            balls.append((np.nan, np.nan) if b.get("x") is None else (float(b["x"]), float(b["y"])))
            rows.append((f, per))
            pos.append(xy)
            det.append(d)
    frame = np.array([f for f, _ in rows], dtype=np.int64)
    period = np.array([p for _, p in rows], dtype=np.int8)
    xy_all = np.stack(pos)
    direction = _infer_home_direction(xy_all, period, players)
    declared = {k + 1: s == "left_to_right" for k, s in enumerate(meta["home_team_side"])}
    for k, v in declared.items():
        if k in direction and direction[k] != v:
            raise ValueError(
                f"{match_id}: inferred home direction {direction} contradicts declared {declared}"
            )
    td = TrackingData(
        match_id=match_id,
        pitch=PitchSpec(float(meta["pitch_length"]), float(meta["pitch_width"])),
        fps=SKILLCORNER_FPS,
        frame=frame,
        period=period,
        time_s=frame / SKILLCORNER_FPS,
        players=players,
        xy=xy_all,
        ball_xy=np.array(balls, dtype=np.float32),
        home_attacks_positive_x=direction,
        provenance=provenance,
        detected=np.stack(det),
    )
    return td, meta


def load_events(events_csv: Path, meta: dict[str, Any], td: TrackingData) -> pl.DataFrame:
    """Map SkillCorner player possessions to the canonical event vocabulary.

    * ``end_type == pass``: successful → ``PASS``, unsuccessful → ``BALL LOST`` (subtype
      ``INTERCEPTION``). Offside and unlabelled outcomes are skipped. ``start_frame`` is the
      **release** (``frame_end`` of the possession). ``end_frame`` is the start of the next
      player possession (reception or interception) when known.
    * ``possession_loss`` → ``BALL LOST``/``THEFT``; ``clearance`` → ``BALL LOST``/``CLEARANCE``;
      ``shot`` → ``SHOT``. All of these are excluded from pass labels by the rules.
    * The pass receiver is the player of the next possession by the same team for a
      successful pass.
    Event x/y are not used (the provider says they are unscaled), so they are NaN here.
    """
    raw = pl.read_csv(events_csv, infer_schema_length=50000)
    _, index, side = _players(meta)
    pid_of = {pid_num: td.players[i].player_id for pid_num, i in index.items()}
    pp = raw.filter(pl.col("event_type") == "player_possession").sort("frame_start")
    recs = pp.select(
        "frame_start",
        "frame_end",
        "period",
        "team_id",
        "player_id",
        "end_type",
        "pass_outcome",
        "time_start",
        "time_end",
    ).to_dicts()
    frames = set(td.frame.tolist())
    out = []
    for k, r in enumerate(recs):
        nxt = recs[k + 1] if k + 1 < len(recs) else None
        et, po = r["end_type"], r["pass_outcome"]
        team = side.get(r["team_id"])
        if team is None:
            continue
        typ: str | None
        sub: str | None = None
        to = None
        if et == "pass" and po == "successful":
            typ = "PASS"
            if nxt is not None and side.get(nxt["team_id"]) == team:
                to = pid_of.get(int(nxt["player_id"]))
        elif et == "pass" and po == "unsuccessful":
            typ, sub = "BALL LOST", "INTERCEPTION"
        elif et == "possession_loss":
            typ, sub = "BALL LOST", "THEFT"
        elif et == "clearance":
            typ, sub = "BALL LOST", "CLEARANCE"
        elif et == "shot":
            typ = "SHOT"
        else:
            continue
        f0 = int(r["frame_end"])
        if f0 not in frames:
            continue
        out.append(
            {
                "event_id": k,
                "team": team,
                "type": typ,
                "subtype": sub,
                "period": int(r["period"]),
                "start_frame": f0,
                "end_frame": int(nxt["frame_start"]) if nxt is not None else None,
                "start_time_s": f0 / SKILLCORNER_FPS,
                "end_time_s": None,
                "from_player_id": pid_of.get(int(r["player_id"])),
                "to_player_id": to,
                "start_x": np.nan,
                "start_y": np.nan,
                "end_x": np.nan,
                "end_y": np.nan,
            }
        )
    return pl.DataFrame(out, schema=EVENT_SCHEMA)
