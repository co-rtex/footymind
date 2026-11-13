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
against goalkeeper positions, and a contradiction raises. The x *and* y sign conventions
were verified independently: event positions (attacking-team frame) rotated by the attack
sign coincide with the tracked passer in all 20 matches (independent review, 2026-11-13).
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


INTERRUPTION_OUT = ("throw_in", "corner", "goal_kick")


def load_events(events_csv: Path, meta: dict[str, Any], td: TrackingData) -> pl.DataFrame:
    """Map SkillCorner player possessions to the canonical event vocabulary (adapter v2).

    Per ``player_possession`` row, in this order:

    * ``RECOVERY`` at ``frame_start`` when ``start_type`` is a recovery or an interception
      (and the possession is not a restart).
    * ``SET PIECE`` immediately before the release when ``game_interruption_before`` is a
      restart awarded to this team (``*_for``). It is placed in the release frame, as Metrica
      encodes set-piece passes, so the restart flag and possession splitting behave the same.
    * The end event at ``frame_end`` (the release):
      - ``end_type == pass``: ``successful`` → ``PASS``. The receiver is ``player_targeted_id``,
        and a ``goal_kick_for`` restart gives subtype ``GOAL KICK``. ``unsuccessful`` →
        ``BALL OUT`` if play stopped for a throw-in, corner or goal kick
        (``game_interruption_after``), ``BALL LOST`` with no subtype for a free kick, otherwise
        ``BALL LOST``/``INTERCEPTION``. Offside and unlabelled outcomes are skipped.
      - ``possession_loss`` → ``BALL LOST``/``THEFT``; ``clearance`` → ``BALL LOST``/``CLEARANCE``;
        ``shot`` → ``SHOT``. None of these are pass attempts under the label rules.

    ``end_frame`` is the start of the next player possession (reception or interception).
    Event positions come from ``x_end``/``y_end``, the passer at ``frame_end``. These are
    metres in the attacking team's frame (verified: after the attack sign they match the
    tracked passer, with a median distance of 0 m). They are rotated into the canonical frame.
    Mapping limitations: SkillCorner has no duel events (``CHALLENGE`` is never produced),
    and the restart mapping is approximate (documented in docs/DATA_SOURCES.md).
    """
    raw = pl.read_csv(events_csv, infer_schema_length=50000)
    _, index, side = _players(meta)
    pid_of = {pid_num: td.players[i].player_id for pid_num, i in index.items()}
    pp = raw.filter(pl.col("event_type") == "player_possession").sort("frame_start")
    cols = [
        "frame_start",
        "frame_end",
        "period",
        "team_id",
        "player_id",
        "end_type",
        "pass_outcome",
        "player_targeted_id",
        "start_type",
        "game_interruption_before",
        "game_interruption_after",
        "x_end",
        "y_end",
    ]
    recs = pp.select([c for c in cols if c in pp.columns]).to_dicts()
    frames = set(td.frame.tolist())
    out: list[dict[str, Any]] = []

    def emit(**kw: Any) -> None:
        blank = {"end_time_s": None, "end_x": np.nan, "end_y": np.nan}
        out.append({"event_id": len(out), **blank, **kw})

    for k, r in enumerate(recs):
        nxt = recs[k + 1] if k + 1 < len(recs) else None
        team = side.get(r["team_id"])
        if team is None:
            continue
        f0, fs = int(r["frame_end"]), int(r["frame_start"])
        if f0 not in frames:
            continue
        per = int(r["period"])
        sign = td.attack_sign(team, per)
        x, y = r.get("x_end"), r.get("y_end")
        sx, sy = (np.nan, np.nan) if x is None or y is None else (float(x) * sign, float(y) * sign)
        player = pid_of.get(int(r["player_id"]))
        gib = r.get("game_interruption_before") or ""
        gia = r.get("game_interruption_after") or ""
        start_type = r.get("start_type") or ""
        restart = gib.endswith("_for")
        base = {"team": team, "period": per, "from_player_id": player}
        won_ball = start_type == "recovery" or "interception" in start_type
        if not restart and won_ball and fs in frames:
            emit(
                type="RECOVERY",
                subtype=None,
                start_frame=fs,
                end_frame=None,
                start_time_s=fs / SKILLCORNER_FPS,
                to_player_id=None,
                start_x=np.nan,
                start_y=np.nan,
                **base,
            )
        et, po = r["end_type"], r["pass_outcome"]
        sub: str | None = None
        to = None
        if et == "pass" and po == "successful":
            typ = "PASS"
            tgt = r.get("player_targeted_id")
            to = pid_of.get(int(tgt)) if tgt is not None else None
            if gib == "goal_kick_for":
                sub = "GOAL KICK"
        elif et == "pass" and po == "unsuccessful":
            if gia.startswith(INTERRUPTION_OUT):
                typ = "BALL OUT"
            elif gia.startswith("free_kick"):
                typ = "BALL LOST"
            else:
                typ, sub = "BALL LOST", "INTERCEPTION"
        elif et == "possession_loss":
            typ, sub = "BALL LOST", "THEFT"
        elif et == "clearance":
            typ, sub = "BALL LOST", "CLEARANCE"
        elif et == "shot":
            typ = "SHOT"
        else:
            continue
        if restart:
            emit(
                type="SET PIECE",
                subtype=gib.removesuffix("_for").replace("_", " ").upper(),
                start_frame=f0,
                end_frame=None,
                start_time_s=f0 / SKILLCORNER_FPS,
                to_player_id=None,
                start_x=sx,
                start_y=sy,
                **base,
            )
        emit(
            type=typ,
            subtype=sub,
            start_frame=f0,
            end_frame=int(nxt["frame_start"]) if nxt is not None else None,
            start_time_s=f0 / SKILLCORNER_FPS,
            to_player_id=to,
            start_x=sx,
            start_y=sy,
            **base,
        )
    return pl.DataFrame(out, schema=EVENT_SCHEMA)
