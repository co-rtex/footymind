"""Derived kinematics (velocity) from positional tracking.

Two estimators, because their correct use differs:

* ``causal``: uses only frames at or before ``t``. Required for any model
  feature evaluated at a decision time, because a centred filter would read up
  to ``window/2`` frames into the future, which would leak the post-decision
  movement into the feature.
* ``centered``: Savitzky-Golay derivative, smoother, for visualisation and
  descriptive analysis only.

Velocities are NaN whenever any frame inside the estimator's window is
unobserved. Gaps are never bridged by interpolation.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_coeffs

MAX_PLAUSIBLE_SPEED = 12.0  # m/s. Elite sprinters peak around 10-11 m/s in matches.


def velocity(
    xy: npt.NDArray[np.floating],
    fps: float,
    method: Literal["causal", "centered"] = "causal",
    window_s: float = 0.2,
    period: npt.NDArray[np.integer] | None = None,
) -> npt.NDArray[np.float32]:
    """Velocity in m/s for positions shaped ``(T, ..., 2)``.

    ``causal``: ``(x[t] - x[t-k]) / (k*dt)`` with ``k = round(window_s*fps)``.
    The result is a backward difference, i.e. the mean velocity over the last
    ``window_s`` seconds.

    ``centered``: Savitzky-Golay first derivative (polyorder 2) over an odd
    window of about ``window_s`` seconds.

    If ``period`` is given, windows never span a period boundary.
    """
    x = np.asarray(xy, dtype=np.float64)
    t = x.shape[0]
    out = np.full(x.shape, np.nan, dtype=np.float64)
    dt = 1.0 / fps
    if method == "causal":
        k = max(1, round(window_s * fps))
        if t > k:
            out[k:] = (x[k:] - x[:-k]) / (k * dt)
            if period is not None:
                p = np.asarray(period)
                cross = p[k:] != p[:-k]
                out[k:][cross] = np.nan
    elif method == "centered":
        w = max(5, round(window_s * fps) | 1)
        half = w // 2
        coeffs = savgol_coeffs(w, polyorder=2, deriv=1, delta=dt, use="dot")
        if t >= w:
            # Sliding-window dot product. NaN anywhere in the window propagates.
            from numpy.lib.stride_tricks import sliding_window_view

            win = sliding_window_view(x, w, axis=0)  # (T-w+1, ..., 2, w)
            out[half : t - half] = win @ coeffs
            if period is not None:
                p = np.asarray(period)
                pw = sliding_window_view(p, w)
                bad = pw.min(axis=1) != pw.max(axis=1)
                out[half : t - half][bad] = np.nan
    else:
        raise ValueError(method)
    return out.astype(np.float32)


def speed(v: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    return np.linalg.norm(np.asarray(v, dtype=np.float64), axis=-1).astype(np.float32)


def implausible_motion_mask(
    xy: npt.NDArray[np.floating],
    fps: float,
    max_speed: float = MAX_PLAUSIBLE_SPEED,
    period: npt.NDArray[np.integer] | None = None,
) -> npt.NDArray[np.bool_]:
    """Frames whose implied single-step speed into or out of them exceeds ``max_speed``.

    Shape ``(T, ...)``, matching ``xy`` without its last axis. Metrica sample
    tracking contains short artefacts: linear drifts at about 20 m/s followed by
    a snap-back, and occasional teleports of tens of metres. Observed
    positions are kept as-is and flagged. Derived quantities should treat the
    flagged frames as unreliable.
    """
    x = np.asarray(xy, dtype=np.float64)
    step = np.linalg.norm(np.diff(x, axis=0), axis=-1) * fps  # (T-1, ...)
    if period is not None:
        p = np.asarray(period)
        step[p[1:] != p[:-1]] = 0.0
    bad_step = np.nan_to_num(step, nan=0.0) > max_speed
    mask = np.zeros(x.shape[:-1], dtype=bool)
    mask[1:] |= bad_step
    mask[:-1] |= bad_step
    return mask


def clean_velocity(
    xy: npt.NDArray[np.floating],
    fps: float,
    method: Literal["causal", "centered"] = "causal",
    window_s: float = 0.2,
    period: npt.NDArray[np.integer] | None = None,
) -> npt.NDArray[np.float32]:
    """``velocity`` with NaN wherever the estimate exceeds the plausible speed limit.

    The window is not inspected for flagged frames. Only the resulting speed is
    checked, which keeps this O(T) and is sufficient for the artefacts observed
    (they inflate the windowed speed well above the limit).
    """
    v = velocity(xy, fps, method, window_s, period)
    s = np.linalg.norm(v, axis=-1)
    v[s > MAX_PLAUSIBLE_SPEED] = np.nan
    return v
