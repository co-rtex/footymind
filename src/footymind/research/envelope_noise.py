"""Tracking-noise and acceleration diagnostics for the kinematic envelope (session 4).

These quantities are estimated **independently of envelope coverage**: nothing here reads
whether a real position fell inside an envelope. They feed the measurement-error margin of
envelope v2 (``counterfactual/kinematics.py``) and answer three diagnostic questions:

1. *Position jitter.* Near-stationary players (a 1.0 s centred window whose least-squares
   line moves slower than ``STATIONARY_MAX_SPEED_MPS``) are tracked with some scatter around
   that line. The residual standard deviation per axis estimates the position noise
   ``sigma_p``. Real micro-movement (shuffling, swaying) is included, so this is an upper
   bound on pure measurement noise (conservative for a margin).
2. *Velocity noise.* In the same windows, the deviation of the causal 0.2 s velocity (the
   one every decision-time feature uses) from the window's line slope estimates the noise of
   that velocity estimator, ``sigma_v`` per axis. At 10 fps the 0.2 s window is two frames.
3. *Acceleration by direction (descriptive).* From a centred (Savitzky-Golay, 0.4 s)
   velocity, the acceleration is split into the component along the direction of motion
   (propulsive if positive, braking if negative) and the normal (turning) component, by speed
   band. Smoothing deflates and noise inflates these values, so they describe the data; they
   are not used to set envelope limits (those are a priori, see the kinematics module).
4. *Velocity lag (descriptive).* The causal 0.2 s velocity is the mean velocity over the last
   0.2 s, i.e. the velocity about 0.1 s earlier. Its deviation from the centred velocity at
   ``t`` (lag plus noise) is compared with its deviation from the centred velocity at
   ``t - 0.1 s`` (noise only). The centred velocity reads future frames, which is acceptable
   for a diagnostic and never used as a feature.

Windows must lie in one period, be complete, contain no implausible-motion artefact and, on
broadcast tracking, consist of *detected* positions only. Sums of squares are returned per
match so that a leave-one-match-out estimate pools the other matches of the same provider.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

from footymind.data.store import load_match
from footymind.spatial.state import causal_velocities
from footymind.tracking.kinematics import implausible_motion_mask, velocity

F = npt.NDArray[np.float64]

STATIONARY_WINDOW_S = 1.0
STATIONARY_MAX_SPEED_MPS = 0.3
CENTRED_VELOCITY_WINDOW_S = 0.4
ACCEL_HALF_SPAN_S = 0.1
SPEED_BANDS_MPS = ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 12.0))
PERCENTILES = (50.0, 95.0, 99.0, 99.9)


def _window_sums(a: F, h: int) -> F:
    """Centred window sums of ``a`` (T, ...) over ``2h + 1`` frames; NaN at the edges."""
    c = np.concatenate([np.zeros((1, *a.shape[1:])), np.cumsum(a, axis=0)], axis=0)
    out = np.full(a.shape, np.nan)
    n = 2 * h + 1
    if a.shape[0] >= n:
        out[h : a.shape[0] - h] = c[n:] - c[:-n]
    return out


def valid_mask(match_id: str) -> tuple[npt.NDArray[np.bool_], Any]:
    """Frames x players whose position is usable for noise estimation."""
    td = load_match(match_id).tracking
    ok = ~np.isnan(td.xy).any(axis=-1)
    ok &= ~implausible_motion_mask(td.xy, td.fps, period=td.period)
    if td.detected is not None:
        ok &= td.detected
    return ok, td


def stationary_noise(match_id: str) -> dict[str, Any]:
    """Sums of squares for ``sigma_p`` and ``sigma_v`` from near-stationary players."""
    ok, td = valid_mask(match_id)
    fps = td.fps
    h = round(STATIONARY_WINDOW_S * fps / 2)
    n = 2 * h + 1
    t_idx = np.arange(td.n_frames, dtype=np.float64)
    xy = np.where(ok[..., None], td.xy.astype(np.float64), 0.0)
    # Centre the coordinates per player to keep the cumulative sums well conditioned.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # players who never appear
        off = np.nan_to_num(np.nanmean(np.where(ok[..., None], td.xy, np.nan), axis=0))
    xy = np.where(ok[..., None], xy - off[None], 0.0)
    good = _window_sums(ok.astype(np.float64), h) == n
    per = td.period.astype(np.float64)
    same_period = _window_sums(per[:, None], h)[:, 0] == n * per
    good &= same_period[:, None]
    s1 = _window_sums(xy, h)
    s2 = _window_sums(xy**2, h)
    st = _window_sums(xy * t_idx[:, None, None], h) - t_idx[:, None, None] * s1
    stt = h * (h + 1) * (2 * h + 1) / 3.0
    slope = st / stt  # metres per frame
    rss = s2 - s1**2 / n - slope**2 * stt  # (T, P, 2)
    line_speed = np.linalg.norm(slope, axis=-1) * fps
    still = good & (line_speed < STATIONARY_MAX_SPEED_MPS)
    vel = causal_velocities(match_id).astype(np.float64)
    dv = vel - slope * fps
    vel_ok = still & ~np.isnan(dv).any(axis=-1)
    rss_s = np.maximum(rss[still], 0.0)
    resid_var = rss_s / (n - 2)  # per window, per axis
    dv_s = dv[vel_ok]
    return {
        "match_id": match_id,
        "fps": fps,
        "window_frames": n,
        "n_windows": int(still.sum()),
        "ss_pos": float(rss_s.sum()),
        "dof_pos": float(2 * (n - 2) * still.sum()),
        "median_sigma_p_m": float(np.sqrt(np.median(resid_var))) if resid_var.size else None,
        "n_vel": int(vel_ok.sum()),
        "ss_vel": float((dv_s**2).sum()),
        "dof_vel": float(2 * vel_ok.sum()),
        "robust_sigma_v_mps": float(1.4826 * np.median(np.abs(dv_s))) if dv_s.size else None,
    }


def pooled_sigmas(parts: list[dict[str, Any]]) -> dict[str, float]:
    """Root-mean-square per-axis ``sigma_p`` and ``sigma_v`` pooled over ``parts``."""
    ss_p = sum(p["ss_pos"] for p in parts)
    dof_p = sum(p["dof_pos"] for p in parts)
    ss_v = sum(p["ss_vel"] for p in parts)
    dof_v = sum(p["dof_vel"] for p in parts)
    return {
        "sigma_p_m": float(np.sqrt(ss_p / dof_p)) if dof_p else float("nan"),
        "sigma_v_mps": float(np.sqrt(ss_v / dof_v)) if dof_v else float("nan"),
    }


def acceleration_profile(match_id: str) -> dict[str, Any]:
    """Histogram-ready acceleration components by speed band (descriptive)."""
    ok, td = valid_mask(match_id)
    fps = td.fps
    vc = velocity(td.xy, fps, "centered", CENTRED_VELOCITY_WINDOW_S, td.period).astype(np.float64)
    j = max(1, round(ACCEL_HALF_SPAN_S * fps))
    acc = np.full(vc.shape, np.nan)
    acc[j:-j] = (vc[2 * j :] - vc[: -2 * j]) / (2 * j / fps)
    same = np.zeros(td.n_frames, dtype=bool)
    same[j:-j] = td.period[2 * j :] == td.period[: -2 * j]
    good = ok & same[:, None] & ~np.isnan(acc).any(axis=-1) & ~np.isnan(vc).any(axis=-1)
    # Usable only if the frames the centred estimates read are usable as well.
    span = round(CENTRED_VELOCITY_WINDOW_S * fps / 2) + j
    okf = ok.astype(np.float64)
    good &= _window_sums(okf, span) == 2 * span + 1
    v = vc[good]
    a = acc[good]
    s = np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = v / s[:, None]
    tang = np.sum(a * u, axis=-1)
    vcaus = causal_velocities(match_id).astype(np.float64)
    lag = round(0.1 * fps)
    vc_lag = np.full(vc.shape, np.nan)
    vc_lag[lag:] = vc[:-lag]
    dv_now = np.linalg.norm(vcaus[good] - v, axis=-1)
    dv_lag = np.linalg.norm(vcaus[good] - vc_lag[good], axis=-1)
    norm = np.abs(a[:, 0] * u[:, 1] - a[:, 1] * u[:, 0])
    out: dict[str, Any] = {"match_id": match_id, "fps": fps, "bands": {}}
    for lo, hi in SPEED_BANDS_MPS:
        m = (s >= lo) & (s < hi)
        band: dict[str, Any] = {"n": int(m.sum())}
        if m.sum() > 100:
            tg, nm = tang[m], norm[m]
            mag = np.linalg.norm(a[m], axis=-1)
            band["propulsive"] = _pct(tg[tg > 0])
            band["braking"] = _pct(-tg[tg < 0])
            band["turning"] = _pct(nm)
            band["share_total_above_3.5"] = float(np.mean(mag > 3.5))
            band["share_total_above_7.0"] = float(np.mean(mag > 7.0))
            band["rms_causal_minus_centred_now_mps"] = _rms(dv_now[m])
            band["rms_causal_minus_centred_0.1s_earlier_mps"] = _rms(dv_lag[m])
        out["bands"][f"{lo:g}-{hi:g}"] = band
    return out


def _rms(x: F) -> float:
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x**2))) if x.size else float("nan")


def _pct(x: F) -> dict[str, float]:
    if x.size == 0:
        return {}
    return {f"p{p:g}": float(np.percentile(x, p)) for p in PERCENTILES}
