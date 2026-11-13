import numpy as np
import pytest

from footymind.tracking.kinematics import (
    clean_velocity,
    implausible_motion_mask,
    velocity,
)

FPS = 25.0


def _line(t: int = 50, v: tuple[float, float] = (3.0, -4.0)) -> np.ndarray:
    k = np.arange(t)[:, None] / FPS
    return (np.array(v)[None, :] * k).astype(np.float32)


@pytest.mark.parametrize("method", ["causal", "centered"])
def test_constant_velocity_recovered(method: str) -> None:
    v = velocity(_line(), FPS, method)  # type: ignore[arg-type]
    finite = np.isfinite(v[:, 0])
    assert finite.sum() > 30
    np.testing.assert_allclose(v[finite], np.tile([3.0, -4.0], (finite.sum(), 1)), atol=1e-3)


def test_causal_velocity_ignores_future() -> None:
    xy = _line()
    base = velocity(xy, FPS, "causal")
    perturbed = xy.copy()
    perturbed[30:] += 100.0  # change the future only
    out = velocity(perturbed, FPS, "causal")
    np.testing.assert_array_equal(base[:30], out[:30])


def test_centered_velocity_reads_future() -> None:
    """Documents why centred velocity must not be used for decision-time features."""
    xy = _line()
    perturbed = xy.copy()
    perturbed[30:] += 1.0
    assert not np.allclose(
        velocity(xy, FPS, "centered")[28], velocity(perturbed, FPS, "centered")[28]
    )


def test_nan_propagates_and_period_boundaries_respected() -> None:
    xy = _line()
    xy[20] = np.nan
    v = velocity(xy, FPS, "causal", window_s=0.2)
    assert np.isnan(v[20]).all() and np.isnan(v[25]).all()  # k=5: frame 25 looks back to 20
    period = np.r_[np.ones(25), 2 * np.ones(25)].astype(np.int8)
    v2 = velocity(_line(), FPS, "causal", period=period)
    assert np.isnan(v2[25:30]).all()
    assert np.isfinite(v2[30]).all()


def test_implausible_motion_flagged_and_cleaned() -> None:
    xy = _line(v=(5.0, 0.0))
    xy[20] += np.array([30.0, 0.0], dtype=np.float32)  # teleport artefact
    mask = implausible_motion_mask(xy, FPS)
    assert mask[19] and mask[20] and mask[21]
    assert mask.sum() == 3
    v = clean_velocity(xy, FPS)
    assert np.isnan(v[20]).all()
    assert np.isfinite(v[40]).all()
