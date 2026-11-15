"""Data-fit short-horizon defender displacement model (the ``learned`` response policy).

For one outfield defender at a decision frame, the model predicts the displacement over the
next ``offset`` seconds from causal, decision-time features, all in the attacking frame
(attacking team towards +x, so the defenders' own goal is at ``x = +L/2``):

* own causal velocity,
* displacement to the ball and the ball's causal velocity,
* displacement to the nearest attacker (a soft minimum over attackers, so the feature is
  continuous when the nearest attacker changes),
* displacement to the own goal centre (goal-side positioning),
* distance to the ball (x model only).

Mirror equivariance by construction. A reflection y -> -y of the attacking frame flips
every y-type feature and leaves the x-type features unchanged. The x displacement uses only
x-type and even features (with an intercept); the y displacement uses only y-type features,
without an intercept, scaling or centring that could break the symmetry. NaN y-type
features are imputed with 0, the symmetric value. A mirrored state therefore gets a
mirrored prediction exactly.

Ridge regression with a fixed penalty (``RIDGE_ALPHA``, chosen a priori). Trained per
provider by ``footymind train-response-model``; serialised as plain JSON (no pickles), like
``spatial/option_model.py``. The model forecasts the *observed* evolution of play. When the
scenario engine applies it to a hypothetical state it extrapolates: no ground truth exists
for responses to interventions.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np
import numpy.typing as npt

F = npt.NDArray[np.float64]

FEATURES_X = ["vx", "ball_dx", "ball_vx", "att_dx", "goal_dx", "ball_dist"]
FEATURES_Y = ["vy", "ball_dy", "ball_vy", "att_dy", "goal_dy"]
SOFT_NEAREST_SIGMA_M = 1.0
RIDGE_ALPHA = 1.0

PROVIDER_RESOURCES = {
    "metrica": "response_model_v1.json",
    "skillcorner": "response_model_v1_skillcorner.json",
}


def soft_nearest_offset(points: F, others: F, sigma: float = SOFT_NEAREST_SIGMA_M) -> F:
    """Softmin-weighted offset from each point to ``others`` (NaN rows ignored)."""
    out = np.full(points.shape, np.nan)
    ok = ~np.isnan(others).any(axis=1)
    if not ok.any():
        return out
    o = others[ok]
    diff = o[None, :, :] - points[:, None, :]  # (n, m, 2)
    d = np.linalg.norm(diff, axis=-1)
    w = np.exp(-(d - d.min(axis=1, keepdims=True)) / sigma)
    w /= w.sum(axis=1, keepdims=True)
    return np.asarray(np.sum(w[..., None] * diff, axis=1))


def features(def_xy: F, def_v: F, ball: F, ball_v: F, att_xy: F, half_length: float) -> tuple[F, F]:
    """Feature matrices ``(n, len(FEATURES_X))`` and ``(n, len(FEATURES_Y))``.

    All inputs in the attacking frame. ``def_*`` have shape ``(n, 2)``, ``att_xy`` ``(m, 2)``.
    """
    to_ball = ball[None, :] - def_xy
    att = soft_nearest_offset(def_xy, att_xy)
    goal = np.array([half_length, 0.0])[None, :] - def_xy
    bv = np.broadcast_to(ball_v, def_xy.shape)
    xx = np.column_stack(
        [
            def_v[:, 0],
            to_ball[:, 0],
            bv[:, 0],
            att[:, 0],
            goal[:, 0],
            np.linalg.norm(to_ball, axis=1),
        ]
    )
    xy = np.column_stack([def_v[:, 1], to_ball[:, 1], bv[:, 1], att[:, 1], goal[:, 1]])
    return xx, xy


@dataclass(frozen=True)
class ResponseModel:
    offsets_s: list[float]  # forecast horizons, one coefficient set each
    x_medians: F
    x_mean: F
    x_scale: F
    coef_x: F  # (H, kx)
    intercept_x: F  # (H,)
    y_scale: F
    coef_y: F  # (H, ky), no intercept (mirror symmetry)
    meta: dict[str, Any]

    @property
    def max_offset_s(self) -> float:
        return max(self.offsets_s)

    def offset_index(self, horizon_s: float) -> int:
        """Index of the trained offset nearest to ``horizon_s`` (capped at the longest)."""
        h = min(horizon_s, self.max_offset_s)
        return int(np.argmin(np.abs(np.asarray(self.offsets_s) - h)))

    def _z(self, xx: F, xy: F) -> tuple[F, F]:
        x = np.where(np.isnan(xx), self.x_medians[None, :], xx)
        zx = (x - self.x_mean[None, :]) / self.x_scale[None, :]
        zy = np.nan_to_num(xy, nan=0.0) / self.y_scale[None, :]
        return zx, zy

    def predict(self, xx: F, xy: F, k: int) -> F:
        """Displacement ``(n, 2)`` in the attacking frame at offset index ``k``."""
        zx, zy = self._z(xx, xy)
        dx = zx @ self.coef_x[k] + self.intercept_x[k]
        dy = zy @ self.coef_y[k]
        return np.column_stack([dx, dy])

    def predict_all(self, xx: F, xy: F) -> F:
        """Displacements ``(n, H, 2)`` for every trained offset."""
        zx, zy = self._z(xx, xy)
        dx = zx @ self.coef_x.T + self.intercept_x[None, :]
        dy = zy @ self.coef_y.T
        return np.stack([dx, dy], axis=-1)

    @classmethod
    def fit(
        cls,
        xx: F,
        xy: F,
        disp: F,
        offsets_s: list[float],
        alpha: float = RIDGE_ALPHA,
        meta: dict[str, Any] | None = None,
    ) -> ResponseModel:
        """Closed-form ridge on standardised features. ``disp`` has shape ``(n, H, 2)``."""
        med = np.nanmedian(xx, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        x = np.where(np.isnan(xx), med[None, :], xx)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale > 1e-9, scale, 1.0)
        zx = (x - mean) / scale
        y0 = np.nan_to_num(xy, nan=0.0)
        y_scale = np.sqrt(np.mean(y0**2, axis=0))
        y_scale = np.where(y_scale > 1e-9, y_scale, 1.0)
        zy = y0 / y_scale
        tx = disp[..., 0]
        ty = disp[..., 1]
        ix = tx.mean(axis=0)
        cx = np.linalg.solve(zx.T @ zx + alpha * np.eye(zx.shape[1]), zx.T @ (tx - ix)).T
        cy = np.linalg.solve(zy.T @ zy + alpha * np.eye(zy.shape[1]), zy.T @ ty).T
        return cls(
            offsets_s=[float(o) for o in offsets_s],
            x_medians=med,
            x_mean=mean,
            x_scale=scale,
            coef_x=np.asarray(cx),
            intercept_x=np.asarray(ix),
            y_scale=y_scale,
            coef_y=np.asarray(cy),
            meta=dict(meta or {}),
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "features_x": FEATURES_X,
                "features_y": FEATURES_Y,
                "soft_nearest_sigma_m": SOFT_NEAREST_SIGMA_M,
                "offsets_s": self.offsets_s,
                "x_medians": self.x_medians.tolist(),
                "x_mean": self.x_mean.tolist(),
                "x_scale": self.x_scale.tolist(),
                "coef_x": self.coef_x.tolist(),
                "intercept_x": self.intercept_x.tolist(),
                "y_scale": self.y_scale.tolist(),
                "coef_y": self.coef_y.tolist(),
                "meta": self.meta,
            },
            indent=1,
        )

    @classmethod
    def from_json(cls, text: str) -> ResponseModel:
        d = json.loads(text)
        if d["features_x"] != FEATURES_X or d["features_y"] != FEATURES_Y:
            raise ValueError("response model was trained with a different feature set")
        a = lambda k: np.asarray(d[k], dtype=np.float64)  # noqa: E731
        return cls(
            offsets_s=[float(o) for o in d["offsets_s"]],
            x_medians=a("x_medians"),
            x_mean=a("x_mean"),
            x_scale=a("x_scale"),
            coef_x=a("coef_x"),
            intercept_x=a("intercept_x"),
            y_scale=a("y_scale"),
            coef_y=a("coef_y"),
            meta=dict(d["meta"]),
        )


_CACHE: dict[tuple[str, str], ResponseModel | None] = {}


def load_response_model(provider: str) -> ResponseModel | None:
    """The packaged model for ``provider``, or None if it has not been trained."""
    resource = PROVIDER_RESOURCES.get(provider)
    if resource is None:
        return None
    key = (resource, provider)  # checked per provider, so one file never serves two
    if key not in _CACHE:
        try:
            text = resources.files("footymind.counterfactual").joinpath(resource).read_text()
            model: ResponseModel | None = ResponseModel.from_json(text)
        except FileNotFoundError:
            model = None
        label = model.meta.get("provider") if model is not None else None
        if model is not None and label != provider:
            # never serve one provider's model on another provider's data (C69)
            warnings.warn(
                f"{resource} is labelled for provider {label!r}, not {provider!r}; not used",
                stacklevel=2,
            )
            model = None
        _CACHE[key] = model
    return _CACHE[key]
