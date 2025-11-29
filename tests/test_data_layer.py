import pandas as pd

from footymind.data.data_loader import load_raw_matches
from footymind.utils.paths import get_raw_data_path


def test_raw_matches_file_exists():
    path = get_raw_data_path("sample_matches.csv")
    assert path.exists(), "sample_matches.csv should exist in data/raw/"


def test_load_raw_matches_basic_columns():
    df = load_raw_matches()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    for col in ["date", "home_team", "away_team", "home_goals", "away_goals"]:
        assert col in df.columns
