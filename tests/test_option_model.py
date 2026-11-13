"""Option model: train/serve feature consistency and JSON round-trip."""

import numpy as np
import polars as pl
import pytest

from footymind.data.store import available_matches, load_match
from footymind.research.pass_dataset import build
from footymind.spatial.option_features import option_feature_rows
from footymind.spatial.option_model import (
    OPTION_FEATURES,
    OPTION_FEATURES_V2,
    OptionModel,
    load_default,
)
from footymind.spatial.state import frame_state

needs_data = pytest.mark.skipif(
    "metrica-sample-1" not in available_matches(), reason="processed Metrica data not available"
)


def test_json_roundtrip_and_nan_imputation() -> None:
    k = len(OPTION_FEATURES)
    m = OptionModel(
        features=OPTION_FEATURES,
        medians=np.ones(k),
        mean=np.zeros(k),
        scale=np.ones(k),
        coef=np.full(k, 0.1),
        intercept=-0.5,
        meta={"name": "synthetic"},
    )
    m2 = OptionModel.from_json(m.to_json())
    row = {f: 1.0 for f in OPTION_FEATURES}
    nan_row = {f: np.nan for f in OPTION_FEATURES}  # imputed with medians (= 1.0)
    p = m2.predict([row, nan_row])
    assert p[0] == pytest.approx(1 / (1 + np.exp(-(0.1 * k - 0.5))))
    assert p[1] == pytest.approx(p[0])


@needs_data
def test_dataset_features_equal_serving_features() -> None:
    """The training row for a pass equals the serving computation for its candidate."""
    mid = "metrica-sample-1"
    df = build([mid]).head(60)
    td = load_match(mid).tracking
    for r in df.iter_rows(named=True):
        s = frame_state(mid, r["frame"])
        ball = s.ball if not np.isnan(s.ball).any() else None
        if ball is None:
            continue
        cand = s.index(r["inferred_receiver_id"])
        passer = td.player_index(r["passer_id"])
        serve = option_feature_rows(s, r["team"], ball, [cand], passer)[0]
        for f in OPTION_FEATURES_V2:
            a, b = r[f], serve[f]
            if a is None or (isinstance(a, float) and np.isnan(a)):
                assert np.isnan(b), f
            else:
                assert b == pytest.approx(a, rel=1e-6, abs=1e-9), f


@needs_data
def test_packaged_model_loads_and_is_documented() -> None:
    m = load_default()
    assert m is not None
    assert m.features == OPTION_FEATURES_V2
    assert m.meta["n_train"] > 1000 and "lomo_pooled" in m.meta
    assert pl.Series(m.coef).is_finite().all()
