"""Train and export the ``learned`` response model (``footymind train-response-model``).

1. Run ``research.response_validation`` for the provider first: it is the honest
   leave-one-match-out evaluation, and its verdict is copied into the model metadata
   (``meta.validation``). The API reads ``validated`` from there.
2. Fit the same specification on every match of the provider and export plain-JSON
   coefficients to ``src/footymind/counterfactual/response_model_v1[_<provider>].json``.

The innovation model (``learned_innovation``, session 4) follows the same pattern with
``research.response_diagnostics`` as its evaluation:
``python -m footymind.research.train_response_model innovation <provider>``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from footymind.counterfactual.response_innovation import (
    INNOVATION_RESOURCES,
    INNOVATION_RIDGE_ALPHA,
    InnovationResponseModel,
)
from footymind.counterfactual.response_model import PROVIDER_RESOURCES, RIDGE_ALPHA, ResponseModel
from footymind.data.store import available_matches, git_revision
from footymind.paths import EXPERIMENTS_DIR
from footymind.research.response_validation import HORIZONS_S, build, fit, result_name

COUNTERFACTUAL_DIR = Path(__file__).resolve().parents[1] / "counterfactual"


def _validation(provider: str, match_ids: list[str]) -> dict[str, Any] | None:
    name = result_name(provider)
    path = EXPERIMENTS_DIR / "results" / name / "results.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    same = sorted(r["matches"]) == sorted(match_ids)
    fde = {
        f"{h:.1f}s": r["summary"][f"{h:.1f}s"]["fde"]["learned_clipped"]["minus_const_velocity"]
        for h in HORIZONS_S
    }
    return {
        # A verdict on a different set of matches does not carry over.
        "validated": bool(r["learned_validated"]) and same,
        "ref": f"experiments/results/{name}/results.json",
        "code_revision": r["code_revision"],
        "same_matches_as_training": same,
        "fde_learned_clipped_minus_const_velocity": {
            h: {"delta_m": v["delta_m"], "ci95": v["ci95"]} for h, v in fde.items()
        },
        "scope": "observational forecasting of real play; not responses to interventions",
    }


def train(
    provider: str, match_ids: list[str] | None = None, out: Path | None = None
) -> ResponseModel:
    if provider not in PROVIDER_RESOURCES:
        raise ValueError(f"no response model registered for provider {provider!r}")
    mids = match_ids or [m for m in available_matches() if m.startswith(provider + "-")]
    if any(not m.startswith(provider + "-") for m in mids):
        raise ValueError("refusing to pool providers; train one provider at a time")
    ds = build(mids)
    meta = {
        "name": "response_model_v1",
        "provider": provider,
        "trained_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_revision": git_revision(),
        "training_matches": mids,
        "n_train": len(ds.p),
        "ridge_alpha": RIDGE_ALPHA,
        "target": "outfield defender displacement in the attacking frame, per frame offset",
        "validation": _validation(provider, mids),
    }
    model = fit(ds, meta=meta)
    out = out or COUNTERFACTUAL_DIR / PROVIDER_RESOURCES[provider]
    out.write_text(model.to_json() + "\n")
    return model


def _innovation_validation(provider: str, match_ids: list[str]) -> dict[str, Any] | None:
    from footymind.research.response_diagnostics import result_name as diag_name

    name = diag_name(provider)
    path = EXPERIMENTS_DIR / "results" / name / "results.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    same = sorted(r["matches"]) == sorted(match_ids)
    d4 = r["d4_innovation_model"]
    return {
        # A verdict on a different set of matches does not carry over.
        "validated": bool(d4["supported"]) and same,
        "ref": f"experiments/results/{name}/results.json",
        "code_revision": r["code_revision"],
        "same_matches_as_training": same,
        "rule": d4["rule"],
        "fde": {
            h: {
                k: {"delta_m": e[k]["mean"], "ci95": e[k]["ci95"]}
                for k in ("innovation_minus_cv", "innovation_minus_placebo")
            }
            for h, e in d4["evaluation"].items()
        },
        "scope": "observational: defender innovation given attackers' real innovations; "
        "not a causal response to interventions",
    }


def train_innovation(
    provider: str, match_ids: list[str] | None = None, out: Path | None = None
) -> InnovationResponseModel:
    """Fit the innovation model on every match of ``provider`` and export plain JSON.

    Run ``research.response_diagnostics`` first: its D4 verdict is copied into the metadata.
    """
    from footymind.research.response_diagnostics import (
        build_features,
        frame_table,
        innovation_target,
    )

    if provider not in INNOVATION_RESOURCES:
        raise ValueError(f"no innovation model registered for provider {provider!r}")
    mids = match_ids or [m for m in available_matches() if m.startswith(provider + "-")]
    if any(not m.startswith(provider + "-") for m in mids):
        raise ValueError("refusing to pool providers; train one provider at a time")
    ds = build(mids)
    tab, row_frame = frame_table(ds)
    feats = build_features(ds, tab, row_frame, None)
    meta = {
        "name": "response_model_innovation_v1",
        "provider": provider,
        "trained_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_revision": git_revision(),
        "training_matches": mids,
        "n_rows": len(ds.p),
        "ridge_alpha": INNOVATION_RIDGE_ALPHA,
        "target": "outfield defender displacement beyond own constant velocity (attacking frame)",
        "validation": _innovation_validation(provider, mids),
    }
    model = InnovationResponseModel.fit(feats, innovation_target(ds), ds.offsets_s, meta=meta)
    out = out or COUNTERFACTUAL_DIR / INNOVATION_RESOURCES[provider]
    out.write_text(model.to_json() + "\n")
    return model


if __name__ == "__main__":
    import sys

    kind, prov = sys.argv[1], sys.argv[2]
    if kind == "innovation":
        train_innovation(prov)
    elif kind == "learned":
        train(prov)
    else:
        raise SystemExit("usage: train_response_model {learned|innovation} <provider>")
