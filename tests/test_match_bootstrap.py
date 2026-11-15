"""Two-stage (match, possession) bootstrap on SYNTHETIC out-of-fold predictions."""

import numpy as np
import polars as pl

from footymind.research.evaluate import match_level_bootstrap, paired_bootstrap


def _oof(n_matches: int, effects: list[float], seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_matches):
        for k in range(300):
            y = int(rng.random() < 0.75)
            pa = 0.75
            # model b moves toward the label by effects[m] (negative effect = worse)
            pb = float(np.clip(pa + effects[m] * (1 if y else -1), 0.05, 0.95))
            rows.append(
                {
                    "match_id": f"m{m}",
                    "possession_id": f"p{k // 3}",
                    "event_id": k,
                    "success": y,
                    "p_a": pa,
                    "p_b": pb,
                }
            )
    return pl.DataFrame(rows)


def test_consistent_gain_is_detected_in_every_match() -> None:
    r = match_level_bootstrap(_oof(6, [0.05] * 6), "a", "b", n_boot=300)
    assert r["log_loss"]["delta"] < 0 and r["log_loss"]["ci95"][1] < 0
    assert r["n_matches_b_better"] == 6 and r["n_matches"] == 6


def test_heterogeneous_matches_widen_the_interval() -> None:
    oof = _oof(6, [0.12, -0.08, 0.10, -0.06, 0.09, -0.07])
    ml = match_level_bootstrap(oof, "a", "b", n_boot=400)
    within = paired_bootstrap(oof, "a", "b", n_boot=400)
    w_ml = ml["log_loss"]["ci95"][1] - ml["log_loss"]["ci95"][0]
    w_in = within["log_loss"]["ci95"][1] - within["log_loss"]["ci95"][0]
    assert w_ml > w_in  # between-match disagreement is reflected only by the two-stage CI
    assert 0 < ml["n_matches_b_better"] < 6
