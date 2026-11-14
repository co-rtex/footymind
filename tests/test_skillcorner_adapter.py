"""SkillCorner adapter tests on a tiny SYNTHETIC fixture (not match observations).

These run in CI, where the real SkillCorner files (~1.8 GB) are not downloaded.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from footymind.tracking.skillcorner import load_events, load_tracking


def _fixture(tmp_path: Path, home_side_p1: str = "left_to_right") -> dict[str, Path]:
    meta = {
        "home_team": {"id": 1},
        "away_team": {"id": 2},
        "home_team_side": [
            home_side_p1,
            "right_to_left" if home_side_p1 == "left_to_right" else "left_to_right",
        ],
        "pitch_length": 105,
        "pitch_width": 68,
        "match_periods": [
            {"period": 1, "start_frame": 10, "end_frame": 13},
            {"period": 2, "start_frame": 20, "end_frame": 23},
        ],
        "players": [
            {"id": 101, "team_id": 1, "number": 1},  # home GK
            {"id": 102, "team_id": 1, "number": 8},
            {"id": 201, "team_id": 2, "number": 1},  # away GK
            {"id": 202, "team_id": 2, "number": 9},
        ],
    }
    lines = [{"frame": f, "period": None, "player_data": [], "ball_data": {}} for f in range(0, 10)]
    for f in list(range(10, 14)) + list(range(20, 24)):
        flip = -1 if f >= 20 else 1
        players = [
            {"player_id": 101, "x": -50.0 * flip, "y": 0.0, "is_detected": True},
            {"player_id": 102, "x": -10.0 * flip, "y": 5.0, "is_detected": f % 2 == 0},
            {"player_id": 201, "x": 50.0 * flip, "y": 0.0, "is_detected": True},
        ]
        if f >= 12:  # 202 only appears from frame 12 (absent before: stays NaN)
            players.append({"player_id": 202, "x": 10.0 * flip, "y": -5.0, "is_detected": False})
        ball = (
            {"x": -9.5 * flip, "y": 5.0, "is_detected": True} if f != 11 else {"x": None, "y": None}
        )
        lines.append(
            {"frame": f, "period": 1 if f < 20 else 2, "player_data": players, "ball_data": ball}
        )
    lines.append({"frame": 30, "period": None, "player_data": [], "ball_data": {}})  # after match
    trk = tmp_path / "t.jsonl"
    trk.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    mj = tmp_path / "m.json"
    mj.write_text(json.dumps(meta))
    ev = tmp_path / "e.csv"
    header = (
        "event_type,frame_start,frame_end,period,team_id,player_id,end_type,pass_outcome,"
        "player_targeted_id,start_type,game_interruption_before,game_interruption_after,x_end,y_end"
    )
    ev.write_text(
        header + "\n"
        # home throw-in restart, completed pass to home_1 (targeted id 101); attacking frame x=10
        "player_possession,10,11,1,1,102,pass,successful,101,throw_in_reception,throw_in_for,,10.0,5.0\n"
        # home GK pass intercepted (in play)
        "player_possession,12,12,1,1,101,pass,unsuccessful,,pass_reception,,,50.0,0.0\n"
        # away wins the ball by interception, then loses it
        "player_possession,13,13,1,2,202,possession_loss,,,pass_interception,,,0.0,0.0\n"
        "passing_option,13,13,1,1,102,,,,,,,,\n"
        # home pass that goes out for a throw-in: a failed attempt recorded as BALL OUT
        "player_possession,20,21,2,1,102,pass,unsuccessful,,pass_reception,,throw_in_against,0.0,0.0\n"
        "player_possession,22,22,2,1,102,pass,offside,202,pass_reception,,,0.0,0.0\n"
    )
    return {"trk": trk, "meta": mj, "ev": ev}


def test_tracking_parse_mask_periods_and_absence(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    td, _ = load_tracking("skillcorner-syn", fx["trk"], fx["meta"], None)
    assert td.frame.tolist() == [10, 11, 12, 13, 20, 21, 22, 23]  # out-of-period frames dropped
    assert td.period.tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert td.fps == 10.0 and td.time_s[0] == pytest.approx(1.0)
    i202 = td.player_index("away_9")
    assert np.isnan(td.xy[:2, i202]).all()  # absent until frame 12: never fabricated
    assert td.detected is not None
    assert not td.detected[2, i202]  # present but extrapolated
    i102 = td.player_index("home_8")
    assert td.detected[0, i102] and not td.detected[1, i102]
    assert np.isnan(td.ball_xy[1]).all()
    # the ball's detection flag is kept: detected where the provider says so, never for NaN
    assert td.ball_detected is not None
    assert td.ball_detected.tolist() == [True, False, True, True, True, True, True, True]
    assert td.home_attacks_positive_x == {1: True, 2: False}


def test_declared_direction_contradiction_raises(tmp_path: Path) -> None:
    fx = _fixture(tmp_path, home_side_p1="right_to_left")
    with pytest.raises(ValueError, match="contradicts declared"):
        load_tracking("skillcorner-syn", fx["trk"], fx["meta"], None)


def test_event_mapping(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    td, meta = load_tracking("skillcorner-syn", fx["trk"], fx["meta"], None)
    ev = load_events(fx["ev"], meta, td)
    got = ev.select(
        "type", "subtype", "start_frame", "end_frame", "from_player_id", "to_player_id"
    ).rows()
    # restart: SET PIECE in the release frame, then the PASS; receiver from player_targeted_id
    assert got[0] == ("SET PIECE", "THROW IN", 11, None, "home_8", None)
    assert got[1] == ("PASS", None, 11, 12, "home_8", "home_1")
    # in-play failure -> interception
    assert got[2] == ("BALL LOST", "INTERCEPTION", 12, 13, "home_1", None)
    # interception start -> RECOVERY at possession start, then the possession loss
    assert got[3][:3] == ("RECOVERY", None, 13)
    assert got[4][:2] == ("BALL LOST", "THEFT")
    # failure ending in a throw-in -> BALL OUT
    assert got[5][:3] == ("BALL OUT", None, 21)
    assert len(got) == 6  # passing_option rows and offside passes are skipped
    # event positions are attacking-frame metres rotated by the attack sign (home attacks +x in P1)
    assert ev["start_x"][1] == pytest.approx(10.0) and ev["start_y"][1] == pytest.approx(5.0)
