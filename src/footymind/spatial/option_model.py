"""Data-calibrated pass-option model ("option model").

A logistic regression predicting completion of a pass *to a given target*, using only
features that can be computed for **any** candidate receiver at a frame. This makes it
applicable to unchosen and hypothetical options in the Possession Multiverse.

Training data: attempted passes (``research.pass_dataset``) with the candidate receiver
inferred from the release direction (78.6% top-1 on completed passes).

Caveat (selection bias): the model is fit on passes that players *chose* to attempt.
Applying it to options they did not choose is extrapolation. Options that are rarely
attempted, such as long backward passes under pressure, may be poorly estimated. The
API labels the output accordingly.

Serialisation: plain JSON (imputer medians, scaler mean/scale, coefficients, intercept,
training metadata). No pickles are used, so loading never executes code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np

OPTION_FEATURES = [
    "ball_x_att",
    "ball_y_abs",
    "dist_to_goal",
    "target_x_att",
    "target_y_abs",
    "cand_distance_m",
    "cand_forward_m",
    "cand_ppcf",
    "cand_p_lane_clear",
    "cand_lane_margin_s",
    "cand_lane_perp_m",
    "cand_def_within_3m_lane",
    "press_nearest_def_m",
    "press_def_within_5m",
    "press_def_within_10m",
    # added after the v0 evaluation (see docs/RESEARCH.md, option model log)
    "target_marker_dist_m",
    "pass_forward_cos",
    "target_speed_along_pass",
]

MODEL_RESOURCE = "option_model_v1.json"


@dataclass(frozen=True)
class OptionModel:
    features: list[str]
    medians: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    meta: dict[str, Any]

    def predict(self, rows: list[dict[str, float]]) -> np.ndarray:
        x = np.array([[r.get(f, np.nan) for f in self.features] for r in rows], dtype=np.float64)
        x = np.where(np.isnan(x), self.medians[None, :], x)
        z = (x - self.mean[None, :]) / self.scale[None, :]
        return 1.0 / (1.0 + np.exp(-(z @ self.coef + self.intercept)))

    def to_json(self) -> str:
        return json.dumps(
            {
                "features": self.features,
                "medians": self.medians.tolist(),
                "mean": self.mean.tolist(),
                "scale": self.scale.tolist(),
                "coef": self.coef.tolist(),
                "intercept": self.intercept,
                "meta": self.meta,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> OptionModel:
        d = json.loads(text)
        return cls(
            features=list(d["features"]),
            medians=np.asarray(d["medians"], dtype=np.float64),
            mean=np.asarray(d["mean"], dtype=np.float64),
            scale=np.asarray(d["scale"], dtype=np.float64),
            coef=np.asarray(d["coef"], dtype=np.float64),
            intercept=float(d["intercept"]),
            meta=dict(d["meta"]),
        )

    @classmethod
    def from_pipeline(cls, pipe: Any, features: list[str], meta: dict[str, Any]) -> OptionModel:
        imp, sc, lr = pipe[0], pipe[1], pipe[2]
        return cls(
            features=features,
            medians=np.asarray(imp.statistics_, dtype=np.float64),
            mean=np.asarray(sc.mean_, dtype=np.float64),
            scale=np.asarray(sc.scale_, dtype=np.float64),
            coef=np.asarray(lr.coef_[0], dtype=np.float64),
            intercept=float(lr.intercept_[0]),
            meta=meta,
        )


_CACHE: dict[str, OptionModel | None] = {}


def load_default() -> OptionModel | None:
    """The packaged option model, or None if it has not been trained yet."""
    if MODEL_RESOURCE not in _CACHE:
        try:
            text = resources.files("footymind.spatial").joinpath(MODEL_RESOURCE).read_text()
            _CACHE[MODEL_RESOURCE] = OptionModel.from_json(text)
        except FileNotFoundError:
            _CACHE[MODEL_RESOURCE] = None
    return _CACHE[MODEL_RESOURCE]
