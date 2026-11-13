"""Metrica Sports sample data adapter, EPTS/FIFA format (Sample Game 3).

Format facts verified against the downloaded files (2026-11-13):

* ``metadata.xml`` (FIFA EPTS). It defines players (``<Player id teamId>`` with
  ``ShirtNumber`` and an optional ``Goalkeeper`` position), teams (``Team A`` =
  ``FIFATMA``, the local/home team per ``Score idLocalTeam``) with
  ``attack_direction_first_half``, the frame rate (25) and one or more
  ``DataFormatSpecification`` blocks. Each block covers a frame range and lists player
  channels in column order. The line format is
  ``frame:p1x,p1y;p2x,p2y;...:ballx,bally``.
* Coordinates are normalised like the CSV format: (0,0) top-left, (1,1)
  bottom-right. Unobserved values are ``NaN``.
* ``events.json``: ``{"data": [...]}``. Subtypes are a list of ``{name}`` objects, or a
  single object, or null. They are joined with "-" in provider order to match the CSV
  vocabulary. Game 3 also contains ``CARRY`` events, which games 1–2 lack.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from footymind.tracking.coords import metrica_to_canonical
from footymind.tracking.metrica import EVENT_SCHEMA, METRICA_PITCH, _infer_home_direction
from footymind.tracking.model import DatasetProvenance, PlayerInfo, TeamSide, TrackingData


@dataclass(frozen=True)
class EptsMeta:
    fps: float
    players: list[PlayerInfo]
    provider_to_index: dict[str, int]  # EPTS player id -> index in players
    team_side: dict[str, TeamSide]  # EPTS team id -> home/away
    specs: list[
        tuple[int, int, list[str]]
    ]  # (start_frame, end_frame, [player ids in column order])
    declared_home_attacks_positive_x_p1: bool | None
    goalkeepers: set[str]  # EPTS player ids declared as goalkeeper


def _text(el: ET.Element | None) -> str | None:
    return el.text.strip() if el is not None and el.text else None


def parse_metadata(path: Path) -> EptsMeta:
    root = ET.parse(path).getroot()
    fps = float(_text(root.find(".//GlobalConfig/FrameRate")) or 25)
    score = root.find(".//Score")
    if score is None:
        raise ValueError("EPTS metadata lacks <Score idLocalTeam=...>")
    local, visiting = score.get("idLocalTeam"), score.get("idVisitingTeam")
    team_side: dict[str, TeamSide] = {local: "home", visiting: "away"}  # type: ignore[dict-item]

    declared: bool | None = None
    for team in root.iter("Team"):
        for pp in team.iter("ProviderParameter"):
            if _text(pp.find("Name")) == "attack_direction_first_half":
                val = _text(pp.find("Value"))
                if team.get("id") == local and val in ("left_to_right", "right_to_left"):
                    declared = val == "left_to_right"

    players: list[PlayerInfo] = []
    index: dict[str, int] = {}
    gks: set[str] = set()
    for p in root.iter("Player"):
        pid, tid = p.get("id"), p.get("teamId")
        if pid is None or tid not in team_side:
            continue
        shirt = int(_text(p.find("ShirtNumber")) or -1)
        side = team_side[tid]
        index[pid] = len(players)
        players.append(PlayerInfo(f"{side}_{shirt}", side, shirt, _text(p.find("Name")) or pid))
        if any(_text(v) == "Goalkeeper" for v in p.iter("Value")):
            gks.add(pid)

    channel_player = {
        ch.get("id"): ch.get("playerId") for ch in root.iter("PlayerChannel") if ch.get("id")
    }
    specs: list[tuple[int, int, list[str]]] = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        order: list[str] = []
        for r in refs[::2]:  # x channel; y follows
            pid = channel_player.get(r)
            if pid is None:
                raise ValueError(f"unknown channel {r}")
            order.append(pid)
        specs.append((int(spec.get("startFrame", 0)), int(spec.get("endFrame", 0)), order))
    if not specs:
        raise ValueError("no DataFormatSpecification in metadata")
    return EptsMeta(fps, players, index, team_side, specs, declared, gks)


def load_tracking(
    match_id: str, tracking_txt: Path, metadata_xml: Path, provenance: DatasetProvenance | None
) -> tuple[TrackingData, EptsMeta]:
    """Parse EPTS tracking. Periods come from the metadata half boundaries.

    The attacking direction is inferred from goalkeeper positions, exactly as for
    the CSV format, and then cross-checked against the provider's declared
    ``attack_direction_first_half``. A disagreement raises, because it would
    indicate a coordinate-orientation error.
    """
    meta = parse_metadata(metadata_xml)
    bounds = period_boundaries(metadata_xml)
    lines = tracking_txt.read_text().splitlines()
    t = len(lines)
    p = len(meta.players)
    frames = np.empty(t, dtype=np.int64)
    raw = np.full((t, p, 2), np.nan)
    ball = np.full((t, 2), np.nan)
    specs = sorted(meta.specs)
    cols = [np.array([meta.provider_to_index[pid] for pid in order]) for _, _, order in specs]
    si = 0
    for row, line in enumerate(lines):
        f_str, players_str, ball_str = line.split(":")
        f = int(f_str)
        frames[row] = f
        while si + 1 < len(specs) and f >= specs[si + 1][0]:
            si += 1
        vals = np.array(players_str.replace(";", ",").split(","), dtype=np.float64)
        raw[row, cols[si]] = vals.reshape(-1, 2)
        bx, by = ball_str.split(",")
        ball[row] = (float(bx), float(by))
    if not np.all(np.diff(frames) > 0):
        raise ValueError("EPTS frames not strictly increasing")
    period = np.zeros(t, dtype=np.int8)
    for k, (a, b) in bounds.items():
        period[(frames >= a) & (frames <= b)] = k
    keep = period > 0  # drop frames outside declared halves (none in Game 3)
    frames, raw, ball, period = frames[keep], raw[keep], ball[keep], period[keep]
    xm, ym = metrica_to_canonical(raw[..., 0], raw[..., 1], METRICA_PITCH)
    bxm, bym = metrica_to_canonical(ball[:, 0], ball[:, 1], METRICA_PITCH)
    xy = np.stack([xm, ym], axis=-1).astype(np.float32)
    direction = _infer_home_direction(xy, period, meta.players)
    if (
        meta.declared_home_attacks_positive_x_p1 is not None
        and direction[1] != meta.declared_home_attacks_positive_x_p1
    ):
        raise ValueError(
            f"inferred home direction {direction[1]} contradicts metadata "
            f"{meta.declared_home_attacks_positive_x_p1}; coordinate orientation error?"
        )
    td = TrackingData(
        match_id=match_id,
        pitch=METRICA_PITCH,
        fps=meta.fps,
        frame=frames,
        period=period,
        time_s=frames / meta.fps,  # matches the CSV convention (frame 1 -> 0.04 s)
        players=meta.players,
        xy=xy,
        ball_xy=np.stack([bxm, bym], axis=-1).astype(np.float32),
        home_attacks_positive_x=direction,
        provenance=provenance,
    )
    return td, meta


def period_boundaries(metadata_xml: Path) -> dict[int, tuple[int, int]]:
    root = ET.parse(metadata_xml).getroot()
    params = {
        _text(pp.find("Name")): _text(pp.find("Value"))
        for pp in root.find(".//GlobalConfig/ProviderGlobalParameters").iter("ProviderParameter")  # type: ignore[union-attr]
    }
    out = {}
    for k, name in (
        (1, "first_half"),
        (2, "second_half"),
        (3, "first_extra_half"),
        (4, "second_extra_half"),
    ):
        a, b = params.get(f"{name}_start"), params.get(f"{name}_end")
        if a and b:
            out[k] = (int(a), int(b))
    return out


def _pid(pid_map: dict[str, str], ref: dict[str, Any] | None) -> str | None:
    key = (ref or {}).get("id")
    return pid_map.get(key) if isinstance(key, str) else None


def load_events(events_json: Path, meta: EptsMeta) -> pl.DataFrame:
    data: list[dict[str, Any]] = json.loads(events_json.read_text())["data"]
    pid_map = {pid: meta.players[i].player_id for pid, i in meta.provider_to_index.items()}
    rows = []
    for k, e in enumerate(data):
        subs = e.get("subtypes")
        if isinstance(subs, dict):
            subs = [subs]
        subtype = "-".join(s["name"] for s in subs) if subs else None
        st, en = e.get("start") or {}, e.get("end") or {}
        sx, sy = metrica_to_canonical(
            st.get("x", np.nan) if st.get("x") is not None else np.nan,
            st.get("y", np.nan) if st.get("y") is not None else np.nan,
            METRICA_PITCH,
        )
        ex, ey = metrica_to_canonical(
            en.get("x", np.nan) if en.get("x") is not None else np.nan,
            en.get("y", np.nan) if en.get("y") is not None else np.nan,
            METRICA_PITCH,
        )
        end_frame = en.get("frame")
        rows.append(
            {
                "event_id": k,
                "team": meta.team_side[e["team"]["id"]],
                "type": e["type"]["name"],
                "subtype": subtype,
                "period": int(e["period"]),
                "start_frame": int(st["frame"]),
                "end_frame": int(end_frame) if end_frame else None,
                "start_time_s": float(st["time"]),
                "end_time_s": float(en["time"]) if en.get("time") else None,
                "from_player_id": _pid(pid_map, e.get("from")),
                "to_player_id": _pid(pid_map, e.get("to")),
                "start_x": float(sx),
                "start_y": float(sy),
                "end_x": float(ex),
                "end_y": float(ey),
            }
        )
    return pl.DataFrame(rows, schema=EVENT_SCHEMA)
