"""Coordinate transforms between provider conventions and the canonical frame."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from footymind.tracking.model import PitchSpec

ArrayLike = npt.ArrayLike


def metrica_to_canonical(
    x: ArrayLike, y: ArrayLike, pitch: PitchSpec
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Metrica normalised coordinates -> canonical metres.

    Metrica: ``(0, 0)`` is the top-left corner, ``(1, 1)`` the bottom-right,
    ``(0.5, 0.5)`` the centre spot (per the provider README). "Top" is the
    upper touchline when the pitch is drawn with x to the right, so the
    provider's y axis points *down*. The canonical y axis points up, so y is
    negated.

    NaN is preserved.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    return (xa - 0.5) * pitch.length, -(ya - 0.5) * pitch.width


def canonical_to_metrica(
    x: ArrayLike, y: ArrayLike, pitch: PitchSpec
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    return xa / pitch.length + 0.5, -ya / pitch.width + 0.5


def to_attacking_frame(xy: npt.NDArray[np.floating], sign: int) -> npt.NDArray[np.floating]:
    """Rotate canonical coordinates so the given team attacks towards +x.

    ``sign`` is +1 if the team already attacks +x (identity), else -1, which
    applies a 180 degree rotation about the centre spot (x -> -x, y -> -y).
    A rotation, not a reflection, preserves left/right handedness. That keeps
    a team's left wing on its left after the transform.
    """
    if sign not in (1, -1):
        raise ValueError("sign must be +1 or -1")
    return xy * sign
