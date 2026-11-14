"""Innovation response model (the ``learned_innovation`` policy), session 4.

The question it answers is the one an intervention asks: *if attackers move differently from
how they were moving, how do defenders move differently from how they were moving?* Both
sides are measured as **innovations**, displacement beyond the player's own constant
velocity over the horizon (the design of the C37 rule check):

    target   iota_j(T) = x_j(t + T) - x_j(t) - v_j T          (outfield defender j)
    inputs   iota_a(T)  for every outfield attacker a, and iota_ball(T) for the ball

For defender ``j`` the inputs are summarised with distance weights fixed by the geometry at
``t`` (attacking frame):

* ``near``: softmin over attackers (1 m scale), i.e. the defender's nearest attacker;
* ``local``: Gaussian kernel mean over attackers, 5 m scale;
* ``zone``: Gaussian kernel mean over attackers, 15 m scale;
* ``team``: unweighted mean over attackers;
* ``ball``: the ball's innovation.

Each summary is a 2-vector. The x innovation is a ridge regression on the five x components
with an intercept (standardised), the y innovation on the five y components without an
intercept or centring, so a mirrored state gets a mirrored prediction exactly (as in
``response_model.py``). A kernel summary is missing when attackers carrying less than half
of its weight have an innovation (unobserved future, unknown velocity or, on broadcast
tracking, an extrapolated position); such rows are excluded, never imputed.

Using the attackers' *future* movement as an input makes this a **conditional response
model**, not a decision-time forecaster: it is evaluated with the attackers' real
innovations given, and it is applied to a hypothetical relocation by treating the relocation
as the relocated attacker's innovation (and the ball's, when the carrier moves). The
association is observational (attackers and defenders share drivers such as the ball); the
permuted-innovation placebo isolates the part specific to the attacker-defender pairing. It
is not a causal estimate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np
import numpy.typing as npt

F = npt.NDArray[np.float64]

INNOVATION_FEATURES = ["near", "local", "zone", "team", "ball"]
NEAR_SIGMA_M = 1.0
LOCAL_SIGMA_M = 5.0
ZONE_SIGMA_M = 15.0
MIN_WEIGHT_SHARE = 0.5
INNOVATION_RIDGE_ALPHA = 1.0

INNOVATION_RESOURCES = {
    "metrica": "response_model_innovation_v1.json",
    "skillcorner": "response_model_innovation_v1_skillcorner.json",
}


def attacker_weights(def_xy: F, att_xy: F) -> F:
    """Weights ``(4, n, m)`` of attackers for each defender: near, local, zone, team.

    Attackers without a position at ``t`` get weight 0. Rows are not normalised: a feature
    normalises over the attackers whose innovation is available.
    """
    n, m = def_xy.shape[0], att_xy.shape[0]
    on = ~np.isnan(att_xy).any(axis=1)
    w = np.zeros((4, n, m))
    if not on.any() or n == 0:
        return w
    d = np.linalg.norm(att_xy[None, on, :] - def_xy[:, None, :], axis=-1)  # (n, m_on)
    w[0][:, on] = np.exp(-(d - d.min(axis=1, keepdims=True)) / NEAR_SIGMA_M)
    w[1][:, on] = np.exp(-0.5 * (d / LOCAL_SIGMA_M) ** 2)
    w[2][:, on] = np.exp(-0.5 * (d / ZONE_SIGMA_M) ** 2)
    w[3][:, on] = 1.0
    return w


def innovation_features(def_xy: F, att_xy: F, att_innov: F, ball_innov: F) -> F:
    """Feature tensor ``(n, K, 5, 2)``; NaN where a summary is missing.

    ``att_innov`` has shape ``(m, K, 2)`` (NaN = unavailable), ``ball_innov`` ``(K, 2)``.
    """
    w = attacker_weights(def_xy, att_xy)  # (4, n, m)
    avail = (~np.isnan(att_innov).any(axis=-1)).astype(np.float64)  # (m, K)
    iv = np.nan_to_num(att_innov)  # NaN entries carry zero weight below
    num = np.einsum("fnm,mk,mkd->nkfd", w, avail, iv)
    den = np.einsum("fnm,mk->nkf", w, avail)
    tot = np.transpose(w.sum(axis=2))[:, None, :]  # (n, 1, 4)
    with np.errstate(invalid="ignore", divide="ignore"):
        att = num / den[..., None]
        share = den / tot
    att[(share < MIN_WEIGHT_SHARE) | (den <= 0)] = np.nan
    k = att_innov.shape[1]
    b = np.asarray(ball_innov, dtype=np.float64).reshape(k, 2)
    ball = np.broadcast_to(b[None, :, None, :], (def_xy.shape[0], k, 1, 2))
    return np.concatenate([att, ball], axis=2)


@dataclass(frozen=True)
class InnovationResponseModel:
    offsets_s: list[float]
    x_mean: F  # (K, 5)
    x_scale: F  # (K, 5)
    coef_x: F  # (K, 5), on standardised x components
    intercept_x: F  # (K,)
    y_scale: F  # (K, 5)
    coef_y: F  # (K, 5), on RMS-scaled y components, no intercept
    meta: dict[str, Any]

    @property
    def max_offset_s(self) -> float:
        return max(self.offsets_s)

    def offset_index(self, horizon_s: float) -> int:
        h = min(horizon_s, self.max_offset_s)
        return int(np.argmin(np.abs(np.asarray(self.offsets_s) - h)))

    def predict(self, feats: F, k: int) -> F:
        """Innovation ``(n, 2)`` at offset index ``k`` from features ``(n, 5, 2)``."""
        zx = (feats[..., 0] - self.x_mean[k]) / self.x_scale[k]
        zy = feats[..., 1] / self.y_scale[k]
        return np.column_stack([zx @ self.coef_x[k] + self.intercept_x[k], zy @ self.coef_y[k]])

    def predict_all(self, feats: F) -> F:
        """Innovations ``(n, K, 2)`` from features ``(n, K, 5, 2)``."""
        zx = (feats[..., 0] - self.x_mean[None]) / self.x_scale[None]
        zy = feats[..., 1] / self.y_scale[None]
        dx = np.einsum("nkf,kf->nk", zx, self.coef_x) + self.intercept_x[None]
        dy = np.einsum("nkf,kf->nk", zy, self.coef_y)
        return np.stack([dx, dy], axis=-1)

    def gains_m_per_m(self) -> dict[str, F]:
        """Coefficients in metres of defender innovation per metre of input, per offset."""
        return {"x": self.coef_x / self.x_scale, "y": self.coef_y / self.y_scale}

    @classmethod
    def fit(
        cls,
        feats: F,
        target: F,
        offsets_s: list[float],
        alpha: float = INNOVATION_RIDGE_ALPHA,
        meta: dict[str, Any] | None = None,
    ) -> InnovationResponseModel:
        """Per-offset closed-form ridge. ``feats`` (n, K, 5, 2), ``target`` (n, K, 2).

        Rows with a NaN feature or target at an offset are left out at that offset.
        """
        k_n, nf = len(offsets_s), feats.shape[2]
        xm, xs, cx = np.zeros((k_n, nf)), np.ones((k_n, nf)), np.zeros((k_n, nf))
        ix, ys, cy = np.zeros(k_n), np.ones((k_n, nf)), np.zeros((k_n, nf))
        n_used = []
        for k in range(k_n):
            ok = ~np.isnan(feats[:, k]).any(axis=(1, 2)) & ~np.isnan(target[:, k]).any(axis=1)
            n_used.append(int(ok.sum()))
            if ok.sum() < 2 * nf:
                continue
            fx, fy = feats[ok, k, :, 0], feats[ok, k, :, 1]
            tx, ty = target[ok, k, 0], target[ok, k, 1]
            xm[k] = fx.mean(axis=0)
            s = fx.std(axis=0)
            xs[k] = np.where(s > 1e-9, s, 1.0)
            zx = (fx - xm[k]) / xs[k]
            ix[k] = tx.mean()
            cx[k] = np.linalg.solve(zx.T @ zx + alpha * np.eye(nf), zx.T @ (tx - ix[k]))
            r = np.sqrt(np.mean(fy**2, axis=0))
            ys[k] = np.where(r > 1e-9, r, 1.0)
            zy = fy / ys[k]
            cy[k] = np.linalg.solve(zy.T @ zy + alpha * np.eye(nf), zy.T @ ty)
        m = dict(meta or {})
        m.setdefault("n_used_per_offset", n_used)
        return cls([float(o) for o in offsets_s], xm, xs, cx, ix, ys, cy, m)

    def to_json(self) -> str:
        return json.dumps(
            {
                "features": INNOVATION_FEATURES,
                "sigmas_m": {"near": NEAR_SIGMA_M, "local": LOCAL_SIGMA_M, "zone": ZONE_SIGMA_M},
                "min_weight_share": MIN_WEIGHT_SHARE,
                "offsets_s": self.offsets_s,
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
    def from_json(cls, text: str) -> InnovationResponseModel:
        d = json.loads(text)
        if d["features"] != INNOVATION_FEATURES:
            raise ValueError("innovation model was trained with a different feature set")
        a = lambda k: np.asarray(d[k], dtype=np.float64)  # noqa: E731
        return cls(
            [float(o) for o in d["offsets_s"]],
            a("x_mean"),
            a("x_scale"),
            a("coef_x"),
            a("intercept_x"),
            a("y_scale"),
            a("coef_y"),
            dict(d["meta"]),
        )


_CACHE: dict[str, InnovationResponseModel | None] = {}


def load_innovation_model(provider: str) -> InnovationResponseModel | None:
    """The packaged innovation model for ``provider``, or None if it has not been trained."""
    resource = INNOVATION_RESOURCES.get(provider)
    if resource is None:
        return None
    if resource not in _CACHE:
        try:
            text = resources.files("footymind.counterfactual").joinpath(resource).read_text()
            _CACHE[resource] = InnovationResponseModel.from_json(text)
        except FileNotFoundError:
            _CACHE[resource] = None
    return _CACHE[resource]
