"""EPTS adapter tests on a tiny SYNTHETIC fixture (not match observations)."""

import json
from pathlib import Path

import numpy as np
import pytest

from footymind.tracking.metrica_epts import load_events, load_tracking, parse_metadata

META = """<?xml version="1.0" encoding="utf-8"?>
<main><Metadata>
 <GlobalConfig><FrameRate>25</FrameRate><ProviderGlobalParameters>
  <ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>
  <ProviderParameter><Name>first_half_end</Name><Value>3</Value></ProviderParameter>
  <ProviderParameter><Name>second_half_start</Name><Value>4</Value></ProviderParameter>
  <ProviderParameter><Name>second_half_end</Name><Value>6</Value></ProviderParameter>
 </ProviderGlobalParameters></GlobalConfig>
 <Sessions><Session id="s"><MatchParameters><Score idLocalTeam="A" idVisitingTeam="B"/></MatchParameters></Session></Sessions>
 <Teams>
  <Team id="A"><Name>Team A</Name><ProviderTeamsParameters><ProviderParameter>
   <Name>attack_direction_first_half</Name><Value>{direction}</Value></ProviderParameter></ProviderTeamsParameters></Team>
  <Team id="B"><Name>Team B</Name></Team>
 </Teams>
 <Players>
  <Player id="PA1" teamId="A"><Name>Player 1</Name><ShirtNumber>1</ShirtNumber>
   <ProviderPlayerParameters><ProviderParameter><Name>position_type</Name><Value>Goalkeeper</Value></ProviderParameter></ProviderPlayerParameters></Player>
  <Player id="PA2" teamId="A"><Name>Player 2</Name><ShirtNumber>2</ShirtNumber></Player>
  <Player id="PB9" teamId="B"><Name>Player 9</Name><ShirtNumber>9</ShirtNumber></Player>
  <Player id="PB7" teamId="B"><Name>Player 7</Name><ShirtNumber>7</ShirtNumber></Player>
 </Players>
 <DataFormatSpecifications>
  <DataFormatSpecification endFrame="3" separator=":" startFrame="1"><StringRegister name="frameCount"/>
   <SplitRegister separator=";">
    <SplitRegister separator=","><PlayerChannelRef playerChannelId="a1x"/><PlayerChannelRef playerChannelId="a1y"/></SplitRegister>
    <SplitRegister separator=","><PlayerChannelRef playerChannelId="b9x"/><PlayerChannelRef playerChannelId="b9y"/></SplitRegister>
    <SplitRegister separator=","><PlayerChannelRef playerChannelId="a2x"/><PlayerChannelRef playerChannelId="a2y"/></SplitRegister>
   </SplitRegister></DataFormatSpecification>
  <DataFormatSpecification endFrame="6" separator=":" startFrame="4"><StringRegister name="frameCount"/>
   <SplitRegister separator=";">
    <SplitRegister separator=","><PlayerChannelRef playerChannelId="a1x"/><PlayerChannelRef playerChannelId="a1y"/></SplitRegister>
    <SplitRegister separator=","><PlayerChannelRef playerChannelId="b7x"/><PlayerChannelRef playerChannelId="b7y"/></SplitRegister>
    <SplitRegister separator=","><PlayerChannelRef playerChannelId="a2x"/><PlayerChannelRef playerChannelId="a2y"/></SplitRegister>
   </SplitRegister></DataFormatSpecification>
 </DataFormatSpecifications>
 <Devices><Device><Sensors><Sensor><Channels>
  <PlayerChannel channelId="x" id="a1x" playerId="PA1"/><PlayerChannel channelId="y" id="a1y" playerId="PA1"/>
  <PlayerChannel channelId="x" id="a2x" playerId="PA2"/><PlayerChannel channelId="y" id="a2y" playerId="PA2"/>
  <PlayerChannel channelId="x" id="b9x" playerId="PB9"/><PlayerChannel channelId="y" id="b9y" playerId="PB9"/>
  <PlayerChannel channelId="x" id="b7x" playerId="PB7"/><PlayerChannel channelId="y" id="b7y" playerId="PB7"/>
 </Channels></Sensor></Sensors></Device></Devices>
</Metadata></main>
"""


def _fixture(tmp_path: Path, direction: str = "left_to_right") -> dict[str, Path]:
    meta = tmp_path / "meta.xml"
    meta.write_text(META.format(direction=direction))
    rows = []
    for f in range(1, 7):
        flip = f >= 4  # teams switch ends in period 2

        def X(v: float, flip: bool = flip) -> str:
            return f"{(1 - v) if flip else v:.5f}"

        # GK A at x=0.02 (left), B outfielder deep on the right, A2 midfield
        second = f"{X(0.97)},0.50000" if f <= 3 else f"{X(0.97)},0.40000"
        ball = "NaN,NaN" if f == 2 else f"{X(0.5)},0.50000"
        rows.append(f"{f}:{X(0.02)},0.50000;{second};{X(0.45)},0.30000:{ball}")
    trk = tmp_path / "tracking.txt"
    trk.write_text("\n".join(rows) + "\n")
    ev = tmp_path / "events.json"
    ev.write_text(json.dumps({"data": [
        {"index": 1, "team": {"id": "A"}, "type": {"name": "PASS"}, "subtypes": None,
         "start": {"frame": 1, "time": 0.04, "x": 0.45, "y": 0.3},
         "end": {"frame": 3, "time": 0.12, "x": 0.02, "y": 0.5}, "period": 1,
         "from": {"id": "PA2"}, "to": {"id": "PA1"}},
        {"index": 2, "team": {"id": "A"}, "type": {"name": "BALL LOST"},
         "subtypes": [{"name": "CROSS"}, {"name": "INTERCEPTION"}],
         "start": {"frame": 5, "time": 0.2, "x": 0.5, "y": 0.5},
         "end": {"frame": 6, "time": 0.24, "x": 0.6, "y": 0.5}, "period": 2,
         "from": {"id": "PA2"}, "to": None},
        {"index": 3, "team": {"id": "B"}, "type": {"name": "RECOVERY"},
         "subtypes": {"name": "INTERCEPTION"},
         "start": {"frame": 6, "time": 0.24, "x": None, "y": None},
         "end": {"frame": None, "time": None, "x": None, "y": None}, "period": 2,
         "from": {"id": "PB7"}, "to": None},
    ]}))  # fmt: skip
    return {"meta": meta, "tracking": trk, "events": ev}


def test_metadata_parsing(tmp_path: Path) -> None:
    m = parse_metadata(_fixture(tmp_path)["meta"])
    assert [p.player_id for p in m.players] == ["home_1", "home_2", "away_9", "away_7"]
    assert m.declared_home_attacks_positive_x_p1 is True
    assert len(m.specs) == 2 and m.specs[1][2] == ["PA1", "PB7", "PA2"]
    assert m.goalkeepers == {"PA1"}


def test_tracking_channels_periods_and_nan(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    td, _ = load_tracking("syn3", fx["tracking"], fx["meta"], None)
    assert td.xy.shape == (6, 4, 2)
    i9, i7 = td.player_index("away_9"), td.player_index("away_7")
    # channel spec switch: away_9 only in frames 1-3, away_7 only in 4-6 (substitution)
    assert np.isfinite(td.xy[:3, i9]).all() and np.isnan(td.xy[3:, i9]).all()
    assert np.isnan(td.xy[:3, i7]).all() and np.isfinite(td.xy[3:, i7]).all()
    assert td.period.tolist() == [1, 1, 1, 2, 2, 2]
    assert np.isnan(td.ball_xy[1]).all()
    assert td.home_attacks_positive_x == {1: True, 2: False}
    assert td.time_s[0] == pytest.approx(0.04)


def test_declared_direction_contradiction_raises(tmp_path: Path) -> None:
    fx = _fixture(tmp_path, direction="right_to_left")
    with pytest.raises(ValueError, match="contradicts metadata"):
        load_tracking("syn3", fx["tracking"], fx["meta"], None)


def test_events_subtypes_joined(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    ev = load_events(fx["events"], parse_metadata(fx["meta"]))
    assert ev["subtype"].to_list() == [None, "CROSS-INTERCEPTION", "INTERCEPTION"]
    assert ev["from_player_id"].to_list() == ["home_2", "home_2", "away_7"]
    assert ev["to_player_id"][0] == "home_1"
    assert ev["end_frame"][2] is None
    assert np.isnan(ev["start_x"][2])
