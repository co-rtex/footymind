"""Missingness of time-on-ball with and without censoring at extrapolated ball frames.

``python -m footymind.research.ball_censor_check [provider]`` writes
``experiments/results/ball_censor_missingness_v1[_<provider>]/results.json``.

It builds the pass dataset of the pre-registered SkillCorner re-run
(``temporal_guard_skillcorner_v2_ballcensor``: rules v2, both guards at 0.2 s) twice, with
``DatasetConfig.censor_undetected_ball`` off and on, and reports per match and in total:

* the share of rows with ``passer_control_s`` unknown (NaN) under each setting;
* how many rows changed and in which direction (censoring may only turn a value into NaN;
  any other change is reported as a violation);
* the share of guard frames whose ball position is provider-extrapolated;
* the ``quick_release`` share among rows with a known value under each setting.

Descriptive data-quality check, no model is fitted. The datasets have the same rows by
construction (the flag changes feature values only); this is checked.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
import polars as pl

from footymind.data.store import available_matches, git_revision, load_match
from footymind.paths import EXPERIMENTS_DIR
from footymind.research.pass_dataset import DatasetConfig, build

GUARD_S = 0.2


def _cfg(censor: bool) -> DatasetConfig:
    return DatasetConfig(
        rules="v2",
        include_ball_out=True,
        temporal_offset_s=GUARD_S,
        static_offset_s=GUARD_S,
        censor_undetected_ball=censor,
    )


def compare(off: pl.DataFrame, on: pl.DataFrame) -> dict[str, Any]:
    """Missingness summary of ``passer_control_s`` between two aligned datasets."""
    key = ["match_id", "event_id"]
    if off.select(key).to_dicts() != on.select(key).to_dicts():
        raise ValueError("the two datasets do not have the same rows")
    a = off["passer_control_s"].to_numpy()
    b = on["passer_control_s"].to_numpy()
    na, nb = np.isnan(a), np.isnan(b)
    same = (na & nb) | (~na & ~nb & np.isclose(a, b, equal_nan=False, atol=1e-9))
    newly_nan = ~na & nb
    violations = ~same & ~newly_nan
    qa = off["quick_release"].to_numpy()
    qb = on["quick_release"].to_numpy()
    return {
        "n": int(a.size),
        "nan_share_off": float(na.mean()),
        "nan_share_on": float(nb.mean()),
        "n_nan_off": int(na.sum()),
        "n_nan_on": int(nb.sum()),
        "n_changed": int((~same).sum()),
        "n_newly_censored": int(newly_nan.sum()),
        "n_violations": int(violations.sum()),
        "mean_tob_off_of_newly_censored": float(a[newly_nan].mean()) if newly_nan.any() else None,
        "quick_release_share_off": float(np.nanmean(qa)),
        "quick_release_share_on": float(np.nanmean(qb)),
    }


def main(provider: str = "skillcorner") -> dict[str, Any]:
    t0 = time.time()
    mids = [m for m in available_matches() if m.startswith(provider + "-")]
    per_match: dict[str, Any] = {}
    frames_off, frames_on = [], []
    for mid in mids:
        off, on = build([mid], _cfg(False)), build([mid], _cfg(True))
        td = load_match(mid).tracking
        s = compare(off, on)
        if td.ball_detected is not None:
            k = round(GUARD_S * td.fps)
            idx = np.array([td.index_of_frame(int(f)) - k for f in off["frame"]])
            s["guard_frame_ball_extrapolated_share"] = float((~td.ball_detected[idx]).mean())
        else:
            s["guard_frame_ball_extrapolated_share"] = None
        per_match[mid] = s
        frames_off.append(off.select("match_id", "event_id", "passer_control_s", "quick_release"))
        frames_on.append(on.select("match_id", "event_id", "passer_control_s", "quick_release"))
    total = compare(pl.concat(frames_off), pl.concat(frames_on))
    res: dict[str, Any] = {
        "name": "ball_censor_missingness_v1" + ("" if provider == "metrica" else f"_{provider}"),
        "provider": provider,
        "code_revision": git_revision(),
        "dataset": {"rules": "v2", "static_offset_s": GUARD_S, "temporal_offset_s": GUARD_S},
        "total": total,
        "per_match": per_match,
        "runtime_s": round(time.time() - t0, 1),
    }
    out = EXPERIMENTS_DIR / "results" / res["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


if __name__ == "__main__":
    r = main(sys.argv[1] if len(sys.argv) > 1 else "skillcorner")
    print(json.dumps(r["total"], indent=2))
