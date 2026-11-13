import numpy as np
import pytest

from footymind.tracking.coords import canonical_to_metrica, metrica_to_canonical, to_attacking_frame
from footymind.tracking.model import PitchSpec

P = PitchSpec(105.0, 68.0)


@pytest.mark.parametrize(
    ("mx", "my", "cx", "cy"),
    [
        (0.5, 0.5, 0.0, 0.0),  # centre spot
        (0.0, 0.0, -52.5, 34.0),  # top-left corner -> (-L/2, +W/2), y up
        (1.0, 1.0, 52.5, -34.0),  # bottom-right
        (1.0, 0.5, 52.5, 0.0),  # right goal centre
    ],
)
def test_metrica_landmarks(mx: float, my: float, cx: float, cy: float) -> None:
    x, y = metrica_to_canonical(mx, my, P)
    assert x == pytest.approx(cx)
    assert y == pytest.approx(cy)


def test_roundtrip_and_nan_preserved() -> None:
    rng = np.random.default_rng(0)
    mx, my = rng.uniform(-0.05, 1.05, 100), rng.uniform(-0.05, 1.05, 100)
    mx[3] = np.nan
    x, y = metrica_to_canonical(mx, my, P)
    bx, by = canonical_to_metrica(x, y, P)
    np.testing.assert_allclose(bx, mx)
    np.testing.assert_allclose(by, my)
    assert np.isnan(x[3])


def test_attacking_frame_is_rotation_not_reflection() -> None:
    # A team attacking -x with a player on its own left wing. Facing -x, the
    # left-hand side is -y. After the transform the team faces +x and the
    # player should be on the left, i.e. +y.
    xy = np.array([[10.0, -20.0]])
    out = to_attacking_frame(xy, -1)
    np.testing.assert_allclose(out, [[-10.0, 20.0]])
    np.testing.assert_allclose(to_attacking_frame(xy, 1), xy)
    with pytest.raises(ValueError):
        to_attacking_frame(xy, 0)
