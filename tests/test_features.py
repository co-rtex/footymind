import pandas as pd

from footymind.data.data_loader import load_raw_matches
from footymind.features.feature_builder import (
    FeatureConfig,
    build_match_features,
)
from footymind.config import TARGET_COLUMN


def test_build_match_features_produces_labels_and_features():
    df_raw = load_raw_matches()
    features_df, matches_df = build_match_features(
        df_raw, FeatureConfig(recent_form_window=5, min_matches_for_features=1)
    )

    assert isinstance(features_df, pd.DataFrame)
    assert isinstance(matches_df, pd.DataFrame)

    # With the sample data, we expect at least one feature row
    assert len(features_df) > 0
    assert TARGET_COLUMN in features_df.columns

    # All feature columns should be numeric except the label
    non_label_cols = [c for c in features_df.columns if c != TARGET_COLUMN]
    assert all(
        pd.api.types.is_numeric_dtype(features_df[c]) for c in non_label_cols
    )
