"""Evidence-grounded tactical explanations (deterministic, no language model).

Every statement is generated from one computed metric and carries that metric's name,
value, provenance and the player ids it refers to. A statement is only made when its
metric exists: no metric, no claim. The thresholds below are descriptive cut-offs chosen
for readability. They are not fitted, and the statements are descriptions of the model's
inputs and outputs, not judgements of what a player should have done.

Statement kinds:

* ``pressure``: the carrier's immediate pressure (DERIVED from positions and velocities);
* ``trend``: how the situation developed over the last second (DERIVED; causal, see
  ``spatial/temporal.py``);
* ``option``: passing options under the option model (ESTIMATED);
* ``caveat``: data-quality and modelling limits that apply to this frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from footymind.spatial.analysis import FrameAnalysis, PassOption
from footymind.spatial.temporal import MAX_CONTROL_S, OPEN_LANE_PERP_M
from footymind.tracking.model import Provenance

UNDER_PRESSURE_M = 3.0
CLOSING_MPS = 1.0
TREND_M = 1.0
PROGRESSIVE_M = 5.0
LIKELY_P = 0.5
UNLIKELY_P = 0.3


@dataclass(frozen=True)
class Statement:
    kind: str
    text: str
    metric: str
    value: float | int | str | None
    provenance: Provenance
    players: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "metric": self.metric,
            "value": self.value,
            "provenance": self.provenance.value,
            "players": self.players,
        }


def _p(o: PassOption) -> float:
    return o.p_success_model if o.p_success_model is not None else o.p_success


def _metric(o: PassOption) -> str:
    return (
        "pass_options.p_success_model"
        if o.p_success_model is not None
        else ("pass_options.p_success (uncalibrated physics; no fitted model)")
    )


def _num(d: dict[str, float], key: str) -> float | None:
    v = d.get(key)
    return float(v) if v is not None and np.isfinite(v) else None


def _finite(x: Any) -> bool:
    return x is not None and isinstance(x, (int, float)) and bool(np.isfinite(x))


def explain(a: FrameAnalysis, temporal: dict[str, float] | None = None) -> list[Statement]:
    """Statements for one analysed frame. ``temporal``: output of ``temporal_context``."""
    st = a.state
    out: list[Statement] = []
    carrier = st.player_ids[st.carrier] if st.carrier is not None else None
    who = carrier or "the carrier"

    def prov_of(ids: list[str | None]) -> Provenance:
        """ESTIMATED if the ball or any referenced player is a provider extrapolation."""
        if st.ball_provenance == Provenance.ESTIMATED:
            return Provenance.ESTIMATED
        for pid in ids:
            if (
                pid
                and st.position_provenance
                and pid in st.player_ids
                and st.position_provenance[st.index(pid)] == Provenance.ESTIMATED
            ):
                return Provenance.ESTIMATED
        return Provenance.DERIVED

    carrier_prov = prov_of([carrier])
    pr = a.pressure
    if pr is not None and _finite(pr.nearest_defender_distance_m):
        d = pr.nearest_defender_distance_m
        tight = d < UNDER_PRESSURE_M or pr.defenders_within_5m >= 2
        text = (
            f"{who} is under pressure: {pr.defenders_within_5m} defender(s) within 5 m, "
            f"nearest {d:.1f} m away"
            if tight
            else f"{who} has space: nearest defender {d:.1f} m away, "
            f"{pr.defenders_within_5m} within 5 m"
        )
        if _finite(pr.closing_speed_mps) and pr.closing_speed_mps > CLOSING_MPS:
            text += f", closing at {pr.closing_speed_mps:.1f} m/s"
        out.append(
            Statement(
                "pressure",
                text + ".",
                "pressure.nearest_defender_distance_m",
                round(d, 2),
                prov_of([carrier, pr.nearest_defender_id]),
                [p for p in (carrier, pr.nearest_defender_id) if p],
            )
        )
    if temporal:
        tob = _num(temporal, "passer_control_s")
        if tob is not None:
            out.append(
                Statement(
                    "trend",
                    (
                        f"The ball has been within 2 m of {who} for at least {tob:.0f} s."
                        if tob >= MAX_CONTROL_S
                        else f"The ball has been within 2 m of {who} for {tob:.1f} s."
                    ),
                    "temporal.passer_control_s",
                    round(tob, 2),
                    carrier_prov,
                    [carrier] if carrier else [],
                )
            )
        ch = _num(temporal, "press_nearest_def_change_m_l10")
        if ch is not None and abs(ch) >= TREND_M:
            trend = "shrank" if ch < 0 else "grew"
            mood = "Pressure is building." if ch < 0 else "Pressure is easing."
            out.append(
                Statement(
                    "trend",
                    f"Over the last second the gap to the nearest defender {trend} by "
                    f"{abs(ch):.1f} m. {mood}",
                    "temporal.press_nearest_def_change_m_l10",
                    round(ch, 2),
                    carrier_prov,
                    [carrier] if carrier else [],
                )
            )
        lanes = _num(temporal, "open_lanes_change_l10")
        if lanes is not None and lanes != 0:
            n = int(lanes)
            out.append(
                Statement(
                    "trend",
                    f"{abs(n)} {'more' if n > 0 else 'fewer'} teammate(s) than a second ago "
                    f"have a straight lane with no defender within {OPEN_LANE_PERP_M:.0f} m.",
                    "temporal.open_lanes_change_l10",
                    n,
                    Provenance.DERIVED,
                )
            )
    opts = [o for o in a.pass_options if _finite(_p(o))]
    if opts:
        safest = max(opts, key=_p)
        out.append(
            Statement(
                "option",
                f"Safest option: {safest.receiver_id} (estimated completion {_p(safest):.2f}, "
                f"{safest.distance_m:.0f} m).",
                _metric(safest),
                round(_p(safest), 3),
                Provenance.ESTIMATED,
                [safest.receiver_id],
            )
        )
        prog = [o for o in opts if o.forward_progress_m > PROGRESSIVE_M]
        if prog:
            best = max(prog, key=_p)
            text = (
                f"Best progressive option: {best.receiver_id}, {best.forward_progress_m:.0f} m "
                f"forward, estimated completion {_p(best):.2f}"
            )
            if best.lane_closest_defender_id and _finite(best.lane_closest_defender_perp_m):
                text += (
                    f"; {best.lane_closest_defender_id} is "
                    f"{best.lane_closest_defender_perp_m:.1f} m from the lane"
                )
            if _p(best) < UNLIKELY_P:
                text += ". No progressive pass is likely to be completed"
            players = [best.receiver_id] + (
                [best.lane_closest_defender_id] if best.lane_closest_defender_id else []
            )
            out.append(
                Statement(
                    "option",
                    text + ".",
                    _metric(best) + "[progressive]",
                    round(_p(best), 3),
                    Provenance.ESTIMATED,
                    players,
                )
            )
        else:
            out.append(
                Statement(
                    "option",
                    f"No teammate is more than {PROGRESSIVE_M:.0f} m ahead of the ball.",
                    "pass_options.forward_progress_m",
                    0,
                    Provenance.DERIVED,
                )
            )
        n_likely = sum(_p(o) > LIKELY_P for o in opts)
        out.append(
            Statement(
                "option",
                f"{n_likely} of {len(opts)} passing options are estimated more likely than not "
                "to be completed.",
                "pass_options.count_over_0_5",
                n_likely,
                Provenance.ESTIMATED,
            )
        )
    for w in a.warnings:
        out.append(
            Statement(
                "caveat", w[0].upper() + w[1:] + ".", "analysis.warnings", None, Provenance.DERIVED
            )
        )
    if opts:
        out.append(
            Statement(
                "caveat",
                "Completion estimates come from a statistical model fitted to this provider's "
                "attempted passes. They describe typical outcomes in similar situations, not what "
                "would certainly happen, and options nobody chose are extrapolated from the "
                "passes players did choose to attempt.",
                "model",
                None,
                Provenance.ESTIMATED,
            )
        )
    return out
