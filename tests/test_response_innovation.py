"""Innovation response model on SYNTHETIC data (seeded; not match observations).

A controlled experiment: defenders are generated with a *known* response to attackers'
unexpected movement. The innovation model must recover it; the session-3 feature set, which
sees only positions and velocities at ``t``, cannot respond to a relocation.
"""

from __future__ import annotations

import numpy as np
import pytest

from footymind.counterfactual.response_innovation import (
    INNOVATION_FEATURES,
    InnovationResponseModel,
    attacker_weights,
    innovation_features,
)
from footymind.counterfactual.response_model import ResponseModel, features

OFFS = [0.5, 1.0]
TRUE_GAIN = 0.6  # synthetic truth: a defender follows 60% of its nearest attacker's innovation


def _synthetic(n_frames: int = 300, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    feats, target, xx, xy, disp = [], [], [], [], []
    for _ in range(n_frames):
        att = rng.uniform([-30, -25], [30, 25], (10, 2))
        dfd = att[:8] + rng.normal(0, 1.5, (8, 2))  # each defender near an attacker
        v_d = rng.normal(0, 2, (8, 2))
        iv = rng.normal(0, 1.0, (10, 2, 2)) * np.array(OFFS)[None, :, None]  # (m, K, 2)
        ball = rng.normal(0, 3, (2, 2))
        f = innovation_features(dfd, att, iv, ball)
        near = f[:, :, 0, :]
        t = TRUE_GAIN * near + rng.normal(0, 0.1, near.shape)
        feats.append(f)
        target.append(t)
        a, b = features(dfd, v_d, att[0] + 0.3, np.zeros(2), att, 52.5)
        xx.append(a)
        xy.append(b)
        disp.append(v_d[:, None, :] * np.array(OFFS)[None, :, None] + t)
    return tuple(np.concatenate(x) for x in (feats, target, xx, xy, disp))


def test_weights_and_missing_features() -> None:
    dfd = np.array([[0.0, 0.0]])
    att = np.array([[1.0, 0.0], [20.0, 0.0], [np.nan, np.nan]])
    w = attacker_weights(dfd, att)
    assert w.shape == (4, 1, 3) and (w[:, :, 2] == 0).all()
    assert w[0, 0, 0] == pytest.approx(1.0) and w[0, 0, 1] < 1e-8
    iv = np.zeros((3, 1, 2))
    iv[0, 0] = [np.nan, np.nan]  # the nearest attacker's innovation is unknown
    f = innovation_features(dfd, att, iv, np.zeros((1, 2)))
    assert np.isnan(f[0, 0, 0]).all()  # near: missing, not imputed
    assert np.isfinite(f[0, 0, 3]).all()  # team mean: half the weight is available
    assert f.shape == (1, 1, len(INNOVATION_FEATURES), 2)


def test_recovers_a_known_response_and_v1_features_cannot() -> None:
    feats, target, xx, xy, disp = _synthetic()
    m = InnovationResponseModel.fit(feats, target, OFFS)
    g = m.gains_m_per_m()
    assert g["x"][1, 0] == pytest.approx(TRUE_GAIN, abs=0.05)
    assert g["y"][1, 0] == pytest.approx(TRUE_GAIN, abs=0.05)
    # Relocating an attacker 5 m: the innovation model moves its defender about 3 m.
    att = np.array([[0.0, 0.0], [25.0, 10.0]])
    dfd = np.array([[0.5, 0.0]])
    zero = np.zeros((2, 1, 2))
    moved = zero.copy()
    moved[0, 0] = [5.0, 0.0]
    base = m.predict(innovation_features(dfd, att, zero, np.zeros((1, 2)))[:, 0], 1)
    resp = m.predict(innovation_features(dfd, att, moved, np.zeros((1, 2)))[:, 0], 1)
    assert np.linalg.norm(resp - base) == pytest.approx(5.0 * TRUE_GAIN, rel=0.1)
    # The v1 features see positions at t only; in this synthetic world the response to a
    # relocation is not identifiable from them, and the fitted offsets stay near zero.
    v1 = ResponseModel.fit(xx, xy, disp, OFFS)
    x0, y0 = features(dfd, np.zeros((1, 2)), np.array([0.3, 0.0]), np.zeros(2), att, 52.5)
    att2 = att.copy()
    att2[0] = [5.0, 0.0]
    x1, y1 = features(dfd, np.zeros((1, 2)), np.array([5.3, 0.0]), np.zeros(2), att2, 52.5)
    assert np.linalg.norm(v1.predict(x1, y1, 1) - v1.predict(x0, y0, 1)) < 0.5


def test_mirror_equivariance_and_json_roundtrip() -> None:
    feats, target, *_ = _synthetic(n_frames=60, seed=1)
    m = InnovationResponseModel.fit(feats, target, OFFS, meta={"synthetic": True})
    mirror = feats.copy()
    mirror[..., 1] *= -1
    p, pm = m.predict_all(feats), m.predict_all(mirror)
    np.testing.assert_allclose(pm[..., 0], p[..., 0], atol=1e-12)
    np.testing.assert_allclose(pm[..., 1], -p[..., 1], atol=1e-12)
    m2 = InnovationResponseModel.from_json(m.to_json())
    np.testing.assert_allclose(m2.predict_all(feats), p)
    assert m2.meta["synthetic"] is True and m2.offset_index(5.0) == 1
