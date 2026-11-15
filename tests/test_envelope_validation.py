"""Units and bookkeeping of research.envelope_validation (SYNTHETIC rows, no match data)."""

from __future__ import annotations

import numpy as np

from footymind.research import envelope_validation as ev
from footymind.research import response_validation as rv
from footymind.research.envelope_noise import _window_sums, pooled_sigmas


def _ds(n_frames: int = 6, per_frame: int = 4, fps: float = 10.0) -> rv.Dataset:
    rng = np.random.default_rng(0)
    n = n_frames * per_frame
    k = rv.horizon_frames(ev.MAX_HORIZON_S, fps)
    p = rng.normal(0, 10, (n, 2))
    v = rng.normal(0, 2, (n, 2))
    offs = np.arange(1, k + 1) / fps
    traj = p[:, None, :] + v[:, None, :] * offs[None, :, None]
    return rv.Dataset(
        fps=fps,
        offsets_s=list(offs),
        match_id=np.array(["m-1"] * (n // 2) + ["m-2"] * (n - n // 2)),
        group=np.array([f"m|{i // 8}" for i in range(n)]),
        p=p,
        v=v,
        traj=traj,
        xx=np.zeros((n, 6)),
        xy=np.zeros((n, 5)),
        excluded={},
        n_frames=n_frames,
        frame=np.repeat(np.arange(n_frames), per_frame),
    )


def test_horizons_are_seconds_converted_with_fps() -> None:
    assert rv.horizon_frames(0.5, 25.0) == 13
    assert rv.horizon_frames(0.5, 10.0) == 5
    assert rv.horizon_frames(1.5, 10.0) == 15
    assert rv.horizon_frames(1.5, 25.0) == 38


def test_permutation_partner_stays_in_frame_and_is_never_self() -> None:
    ds = _ds()
    partner = ev.permutation_partner(ds, seed=1)
    assert (partner >= 0).all()
    assert (partner != np.arange(len(partner))).all()
    assert (ds.frame[partner] == ds.frame).all() and (ds.match_id[partner] == ds.match_id).all()
    # a frame with a single row has no partner
    ds.frame[0] = 99
    assert ev.permutation_partner(ds, seed=1)[0] == -1


def test_constant_velocity_rows_are_covered_and_shifts_rejected() -> None:
    ds = _ds()
    k = rv.horizon_frames(0.5, ds.fps)
    polys = ev.polygons(ds.p, ds.v, k / ds.fps, ev.FIXED_VARIANTS["v1"])
    shifted = ds.traj[:, k - 1] + np.array([ev.SHIFT_M, 0.0])
    cols = ev._metrics(ds, polys, k, shifted, ev.permutation_partner(ds, 0))
    assert cols["coverage"].mean() > 0.95  # constant velocity is always reachable
    assert cols["shift_rejection"].mean() == 1.0
    # the constant-speed disc: radius speed * t around the start
    disc = ev.polygons(ds.p[:1], ds.v[:1], 1.0, 7.0)[0]
    assert np.linalg.norm(disc - ds.p[0], axis=1).min() >= 7.0 - 1e-9


def test_pooled_sigmas_and_window_sums() -> None:
    parts = [
        {"ss_pos": 4.0, "dof_pos": 100.0, "ss_vel": 1.0, "dof_vel": 100.0},
        {"ss_pos": 0.0, "dof_pos": 300.0, "ss_vel": 3.0, "dof_vel": 300.0},
    ]
    s = pooled_sigmas(parts)
    assert np.isclose(s["sigma_p_m"], 0.1) and np.isclose(s["sigma_v_mps"], 0.1)
    a = np.arange(10, dtype=float)[:, None]
    w = _window_sums(a, 1)
    assert np.isnan(w[0, 0]) and w[1, 0] == 3.0 and np.isnan(w[-1, 0])
