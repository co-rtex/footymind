"""Kinematic envelope validation: v1 (session 3) against the pre-registered v2 (session 4).

``python -m footymind.research.envelope_validation <provider> [max_matches] [--noise-only]``
writes ``experiments/results/envelope_validation_v1[_<provider>]/results.json``.

Question. Does an envelope contain the positions real outfield defenders reach, while still
rejecting positions they could not have reached? Both matter: the envelope clips response
policy targets (a reachable target must not be clipped) and flags user moves that are
physically implausible (an implausible move must not pass).

Rows. The defender rows of ``research.response_validation`` (pass-start frames, outfield
defenders with a causal velocity, no artefact, detected-only on broadcast tracking), with a
complete trajectory to 1.5 s. Horizons are in seconds; frame offsets are
``round(h * fps)`` (13 frames = 0.52 s at 25 Hz, 5 frames at 10 fps).

Pre-registered before any v2 coverage was computed (2026-11-14, session 4)
-------------------------------------------------------------------------
* Envelope v2 is defined in ``counterfactual/kinematics.py`` with a priori limits. Its only
  data-estimated parameters are the per-provider noise levels ``sigma_p`` and ``sigma_v`` from
  near-stationary players (``research/envelope_noise.py``). They are estimated
  leave-one-match-out here: the rows of a held-out match use the noise pooled over the other
  matches of the same provider.
* Metrics per provider and horizon (0.5, 1.0, 1.5 s): *coverage* (share of real positions
  inside), *shift rejection* (share of real positions moved 5 m in a uniformly random
  direction that fall outside), *permutation rejection* (share of other outfield defenders'
  real positions in the same frame that fall outside), mean polygon area, and cost.
  Possession-clustered bootstrap (1,000, stratified by match).
* Adoption rule: v2 becomes the default envelope iff, on **both** providers,
  coverage >= 0.95 at 0.5 s and at 1.0 s, shift rejection >= 0.90 at 0.5 s and >= 0.50 at
  1.0 s, permutation rejection >= 0.90 at 0.5 s and 1.0 s, and constructing one envelope
  takes <= 5 ms (median). Point estimates are compared with the thresholds; CIs are
  reported. The 1.0 s shift threshold is lower because a sprint start covers about 3.3 m in
  1.1 s, so a 5 m offset is near the physical boundary there. 1.5 s is reported, not gated.
* Ablations (descriptive, never adoptable): v1 without the reaction delay; v2 without the
  noise margin, without the velocity lag, without the force-velocity limit (isotropic), and
  with v1's 3.5 m/s^2 bound. They apportion v1's misses among the candidate causes.
* Comparator: the scenario's constant-speed disc (``REACH_SPEED_MPS`` x horizon), which
  decides ``supported``.
* Secondary: the ``learned`` response model's pre-specified forecast rule
  (``response_validation.validated``) re-checked with v2 clipping (LOMO model, 1.0 s rows).

Providers are never pooled.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt

from footymind.counterfactual.kinematics import (
    DEFAULT_LIMITS,
    KinematicLimits,
    KinematicLimitsV2,
    Limits,
    envelope_polygons,
    inside_polygon,
    project_onto_polygon,
    reachability_envelope,
    support_polygon,
)
from footymind.counterfactual.scenario import REACH_SPEED_MPS
from footymind.data.store import available_matches, git_revision
from footymind.paths import EXPERIMENTS_DIR
from footymind.research import response_validation as rv
from footymind.research.envelope_noise import acceleration_profile, pooled_sigmas, stationary_noise

F = npt.NDArray[np.float64]

HORIZONS_S = (0.5, 1.0, 1.5)
MAX_HORIZON_S = 1.5
SHIFT_M = 5.0
N_BOOT = 1000
CRITERIA = {
    "coverage_min": {"0.5s": 0.95, "1.0s": 0.95},
    "shift_rejection_min": {"0.5s": 0.90, "1.0s": 0.50},
    "permutation_rejection_min": {"0.5s": 0.90, "1.0s": 0.90},
    "max_ms_per_envelope": 5.0,
}
_CHUNK = 2000
Variant = Limits | float  # a float is the speed of a constant-speed disc


def result_name(provider: str) -> str:
    return "envelope_validation_v1" + ("" if provider == "metrica" else f"_{provider}")


def v2_variants(sp: float, sv: float) -> dict[str, KinematicLimitsV2]:
    base = KinematicLimitsV2(position_sigma_m=sp, velocity_sigma_mps=sv, noise_source="LOMO")
    return {
        "v2": base,
        "v2_no_margin": replace(base, margin_z=0.0),
        "v2_no_lag": replace(base, velocity_lag_s=0.0),
        "v2_isotropic": replace(base, force_velocity=False),
        "v2_accel_3.5": replace(base, max_accel_mps2=3.5, propulsive_accel_mps2=3.5),
    }


FIXED_VARIANTS: dict[str, Variant] = {
    "v1": DEFAULT_LIMITS,
    "v1_no_delay": replace(DEFAULT_LIMITS, reaction_s=0.0),
    "disc_reach_speed": REACH_SPEED_MPS,
}


def polygons(p: F, v: F, t: float, variant: Variant) -> F:
    """Envelope polygons ``(N, n, 2)`` at horizon ``t`` (seconds) for rows ``p``, ``v``."""
    out = []
    for a in range(0, len(p), _CHUNK):
        sl = slice(a, a + _CHUNK)
        if isinstance(variant, float):
            n = DEFAULT_LIMITS.n_headings
            theta = 2.0 * np.pi * np.arange(n) / n
            u = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
            out.append(support_polygon(p[sl] @ u.T + variant * t, n))
        else:
            out.append(envelope_polygons(p[sl], v[sl], [t], variant)[0][0])
    return np.concatenate(out) if out else np.zeros((0, DEFAULT_LIMITS.n_headings, 2))


def area(poly: F) -> F:
    x, y = poly[..., 0], poly[..., 1]
    return np.asarray(
        0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=-1) - y * np.roll(x, -1, axis=-1), axis=-1))
    )


def permutation_partner(ds: rv.Dataset, seed: int) -> npt.NDArray[np.int64]:
    """For each row, another row of the same decision frame (-1 if it is alone)."""
    rng = np.random.default_rng(seed)
    key = np.char.add(np.char.add(ds.match_id.astype(str), "|"), ds.frame.astype(str))
    _, inv = np.unique(key, return_inverse=True)
    partner = np.full(len(key), -1, dtype=np.int64)
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    for g in np.split(order, bounds):
        if g.size < 2:
            continue
        perm = rng.permutation(g)
        partner[perm] = np.roll(perm, 1)  # a cyclic shift of a random order: never itself
    return partner


def _metrics(
    ds: rv.Dataset, polys: F, k: int, shifted: F, partner: npt.NDArray[np.int64]
) -> dict[str, F]:
    real = ds.traj[:, k - 1]
    perm = np.where(partner[:, None] >= 0, real[np.maximum(partner, 0)], np.nan)
    has = partner >= 0
    rej_perm = np.full(len(real), np.nan)
    rej_perm[has] = (~inside_polygon(polys[has], perm[has])).astype(float)
    return {
        "coverage": inside_polygon(polys, real).astype(float),
        "shift_rejection": (~inside_polygon(polys, shifted)).astype(float),
        "permutation_rejection": rej_perm,
        "area_m2": area(polys),
    }


def _bootstrap(ds: rv.Dataset, cols: dict[str, F], seed: int) -> dict[str, dict[str, Any]]:
    names = list(cols)
    uniq, ginv = np.unique(ds.group, return_inverse=True)
    g_match = np.array([u.split("|")[0] for u in uniq])
    strata = [np.flatnonzero(g_match == m) for m in np.unique(g_match)]
    w = rv._bootstrap_weights(np.arange(uniq.size), strata, N_BOOT, seed)
    out = {}
    for n in names:
        x = cols[n]
        ok = np.isfinite(x)
        sums = np.bincount(ginv[ok], weights=x[ok], minlength=uniq.size)
        cnt = np.bincount(ginv[ok], minlength=uniq.size).astype(float)
        boot = (w @ sums) / np.maximum(w @ cnt, 1e-12)
        out[n] = {
            "mean": float(x[ok].mean()) if ok.any() else None,
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n": int(ok.sum()),
        }
    return out


def noise_parts(mids: list[str]) -> dict[str, dict[str, Any]]:
    return {m: stationary_noise(m) for m in mids}


def lomo_sigmas(parts: dict[str, dict[str, Any]], held_out: str) -> tuple[dict[str, float], bool]:
    others = [v for m, v in parts.items() if m != held_out]
    if not others:  # a single-match pilot cannot hold out; flagged in the output
        return pooled_sigmas([parts[held_out]]), True
    return pooled_sigmas(others), False


def timing() -> dict[str, float]:
    out = {}
    lims: dict[str, Limits] = {
        "v1": DEFAULT_LIMITS,
        "v2": KinematicLimitsV2(position_sigma_m=0.03, velocity_sigma_mps=0.1),
    }
    for name, lim in lims.items():
        ts = []
        for _ in range(200):
            t0 = time.perf_counter()
            reachability_envelope([1.0, 2.0], [3.0, 1.0], 1.5, lim)
            ts.append(time.perf_counter() - t0)
        out[f"{name}_median_ms_per_envelope_1.5s"] = float(np.median(ts) * 1e3)
    return out


def learned_under_v2(
    ds: rv.Dataset, limits_by_match: dict[str, KinematicLimitsV2], seed: int
) -> dict[str, Any]:
    """The learned model's pre-specified forecast rule with v2 clipping (1.0 s rows)."""
    k1 = rv.horizon_frames(rv.MAX_HORIZON_S, ds.fps)
    d1 = replace(ds, traj=ds.traj[:, :k1], offsets_s=ds.offsets_s[:k1])
    preds = rv.predictions(d1)  # LOMO model; v1-clipped variants included
    for name, base in (
        ("learned_clipped", preds["learned"]),
        ("const_velocity_clipped", preds["const_velocity"]),
    ):
        clipped = np.empty_like(base)
        for m, lim in limits_by_match.items():
            te = np.flatnonzero(d1.match_id == m)
            for a in range(0, te.size, _CHUNK):
                idx = te[a : a + _CHUNK]
                verts, _, _ = envelope_polygons(d1.p[idx], d1.v[idx], d1.offsets_s, lim)
                for kk in range(k1):
                    clipped[idx, kk] = project_onto_polygon(verts[kk], base[idx, kk])
        preds[name] = clipped
    summary = rv.evaluate(d1, preds, seed=seed)
    brief = {
        h: {
            "fde_learned_clipped_v2_m": summary[h]["fde"]["learned_clipped"]["mean_m"],
            "fde_const_velocity_m": summary[h]["fde"]["const_velocity"]["mean_m"],
            "fde_learned_unclipped_m": summary[h]["fde"]["learned"]["mean_m"],
            "minus_const_velocity": {
                k: summary[h]["fde"]["learned_clipped"]["minus_const_velocity"][k]
                for k in ("delta_m", "ci95", "better_in_matches", "n_matches")
            },
            "cv_clipped_v2_minus_cv_m": summary[h]["fde"]["const_velocity_clipped"][
                "minus_const_velocity"
            ]["delta_m"],
        }
        for h in ("0.5s", "1.0s")
    }
    return {"rule_met_with_v2_clip": rv.validated(summary), "fde": brief}


def adopt(by_variant: dict[str, Any], timing_ms: float) -> dict[str, Any]:
    """Apply the pre-registered rule to the v2 entry of one provider."""
    checks: dict[str, bool] = {}
    v2 = by_variant["v2"]
    for metric, mins in (
        ("coverage", CRITERIA["coverage_min"]),
        ("shift_rejection", CRITERIA["shift_rejection_min"]),
        ("permutation_rejection", CRITERIA["permutation_rejection_min"]),
    ):
        assert isinstance(mins, dict)
        for h, lo in mins.items():
            checks[f"{metric}@{h}>={lo}"] = bool(v2[h][metric]["mean"] >= lo)
    checks[f"ms_per_envelope<={CRITERIA['max_ms_per_envelope']}"] = bool(
        timing_ms <= float(CRITERIA["max_ms_per_envelope"])  # type: ignore[arg-type]
    )
    return {"checks": checks, "criterion_met": all(checks.values())}


def main(
    provider: str = "metrica",
    max_matches: int | None = None,
    seed: int = 0,
    noise_only: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    mids = [m for m in available_matches() if m.startswith(provider + "-")][:max_matches]
    parts = noise_parts(mids)
    noise = {
        "method": "near-stationary players: 1.0 s centred window, line speed < 0.3 m/s; "
        "per-axis RMS residual (sigma_p) and causal-velocity deviation from the line slope "
        "(sigma_v)",
        "pooled_all_matches": pooled_sigmas(list(parts.values())),
        "per_match": parts,
    }
    if noise_only:
        print(json.dumps(noise["pooled_all_matches"], indent=2))
        return noise
    accel = {m: acceleration_profile(m) for m in mids}
    ds = rv.build(mids, MAX_HORIZON_S)
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 2.0 * np.pi, len(ds.p))
    partner = permutation_partner(ds, seed)
    lomo: dict[str, dict[str, float]] = {}
    in_sample = False
    for m in mids:
        sig, flag = lomo_sigmas(parts, m)
        lomo[m] = sig
        in_sample |= flag
    by_variant: dict[str, dict[str, Any]] = {}
    per_match_cov: dict[str, dict[str, dict[str, float]]] = {}
    throughput: dict[str, float] = {}
    variant_names = [*FIXED_VARIANTS, *v2_variants(0.0, 0.0)]
    for name in variant_names:
        by_variant[name] = {}
        per_match_cov[name] = {}
        tv = time.perf_counter()
        n_env = 0
        for h in HORIZONS_S:
            k = rv.horizon_frames(h, ds.fps)
            t_act = k / ds.fps
            polys = np.empty((len(ds.p), DEFAULT_LIMITS.n_headings, 2))
            for m in mids:
                te = np.flatnonzero(ds.match_id == m)
                var: Variant = (
                    FIXED_VARIANTS[name]
                    if name in FIXED_VARIANTS
                    else v2_variants(lomo[m]["sigma_p_m"], lomo[m]["sigma_v_mps"])[name]
                )
                polys[te] = polygons(ds.p[te], ds.v[te], t_act, var)
            n_env += len(ds.p)
            real = ds.traj[:, k - 1]
            shifted = real + SHIFT_M * np.column_stack([np.cos(phi), np.sin(phi)])
            cols = _metrics(ds, polys, k, shifted, partner)
            hk = f"{h:.1f}s"
            by_variant[name][hk] = {"horizon_frames": k, "horizon_s_actual": t_act}
            by_variant[name][hk] |= _bootstrap(ds, cols, seed)
            per_match_cov[name][hk] = {
                m: float(cols["coverage"][ds.match_id == m].mean()) for m in mids
            }
        throughput[name] = n_env / (time.perf_counter() - tv)
    tim = timing()
    decision = adopt(by_variant, tim["v2_median_ms_per_envelope_1.5s"])
    limits_by_match = {
        m: v2_variants(lomo[m]["sigma_p_m"], lomo[m]["sigma_v_mps"])["v2"] for m in mids
    }
    learned = learned_under_v2(ds, limits_by_match, seed)
    speed = np.linalg.norm(ds.v, axis=1)
    name = result_name(provider)
    result = {
        "name": name,
        "provider": provider,
        "question": "do kinematic envelopes contain real defender positions and reject "
        "implausible ones (v1 vs pre-registered v2)?",
        "matches": mids,
        "code_revision": git_revision(),
        "runtime_s": None,
        "fps": ds.fps,
        "n": len(ds.p),
        "n_decision_frames": ds.n_frames,
        "n_possessions": int(np.unique(ds.group).size),
        "excluded_defender_rows": ds.excluded,
        "rows": "response_validation rows with a complete 1.5 s trajectory",
        "shift_m": SHIFT_M,
        "n_boot": N_BOOT,
        "bootstrap_unit": "possession (stratified by match)",
        "noise": noise,
        "noise_lomo": lomo,
        "noise_in_sample_pilot": in_sample,
        "acceleration_profile": accel,
        "share_rows_speed_above": {
            f"{s:g}": float(np.mean(speed > s)) for s in (7.5, KinematicLimitsV2().max_speed_mps)
        },
        "limits": {
            "v1": _describe(DEFAULT_LIMITS),
            "v2_template": _describe(KinematicLimitsV2()),
            "disc_reach_speed_mps": REACH_SPEED_MPS,
        },
        "criteria": CRITERIA,
        "by_variant": by_variant,
        "per_match_coverage": per_match_cov,
        "timing": tim,
        "throughput_envelopes_per_s": throughput,
        "adoption": decision,
        "learned_with_v2_clip": learned,
    }
    result["runtime_s"] = round(time.time() - t0, 1)
    out = EXPERIMENTS_DIR / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    brief = {
        v: {
            h: {
                k: round(by_variant[v][h][k]["mean"], 3)
                for k in ("coverage", "shift_rejection", "permutation_rejection", "area_m2")
            }
            for h in by_variant[v]
            if h.endswith("s")
        }
        for v in by_variant
    }
    print(json.dumps({"n": len(ds.p), "brief": brief, "adoption": decision}, indent=1))
    return result


def _describe(lim: KinematicLimits | KinematicLimitsV2) -> dict[str, Any]:
    d = {k: getattr(lim, k) for k in lim.__dataclass_fields__}
    d["version"] = lim.version
    d["describe"] = lim.describe()
    return d


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "metrica",
        max_matches=int(args[1]) if len(args) > 1 else None,
        noise_only="--noise-only" in sys.argv,
    )
