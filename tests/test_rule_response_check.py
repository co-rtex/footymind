"""Decision rule of the observational rule-policy check, on SYNTHETIC error rows."""

import polars as pl

from footymind.research.rule_response_check import _summary


def _df(gain: float) -> pl.DataFrame:
    rows = [
        {
            "match_id": "m",
            "possession_id": f"p{i % 40}",
            "policy": "x",
            "move_m": 1.0,
            "err_cv": 1.0,
            "err_policy": 1.0 - gain + 0.01 * ((i % 7) - 3),
        }
        for i in range(200)
    ]
    return pl.DataFrame(rows)


def test_supported_only_when_errors_fall() -> None:
    assert _summary(_df(0.2), 300, 0)["observationally_supported"]
    s = _summary(_df(-0.2), 300, 0)
    assert not s["observationally_supported"] and s["delta_m"] > 0


def test_rule_policies_carry_the_observational_caveat_and_learned_does_not() -> None:
    from footymind.counterfactual.response import POLICIES

    flagged = {n for n, sp in POLICIES.items() if any("C37" in a for a in sp.assumptions)}
    assert flagged == {"marker_follows", "press_carrier", "zonal_shift", "lane_cover"}


def test_paired_contrast_detects_a_better_design() -> None:
    """SYNTHETIC rows: the 'innovation' design improves on CV more than the placebo does."""
    from footymind.research.rule_response_check import _paired

    rows = []
    for i in range(120):
        for design, gain in (("innovation", -0.10), ("placebo_rot", -0.02)):
            rows.append(
                {
                    "match_id": f"m{i % 3}",
                    "possession_id": f"p{i // 2}",
                    "frame": i,
                    "design": design,
                    "policy": "x",
                    "move_m": 1.0,
                    "err_cv": 1.0,
                    "err_policy": 1.0 + gain + 0.01 * ((i % 5) - 2),
                }
            )
    r = _paired(pl.DataFrame(rows), "innovation", "placebo_rot", 300, 0)
    assert r is not None and r["delta_m"] < 0 and r["ci95"][1] < 0
