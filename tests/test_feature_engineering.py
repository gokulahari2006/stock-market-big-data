"""Unit tests for feature engineering and target leakage prevention."""

import pytest
import pandas as pd
import numpy as np

from src.feature_engineering import FeatureEngineer, FeatureEngineeringError
from src.config import FEATURE_COLUMNS, TARGET_COLUMN


@pytest.fixture
def synthetic_ohlcv():
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    prices = [100.0 + i * 2.0 for i in range(100)]
    return pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": [p + 2.0 for p in prices],
        "low": [p - 2.0 for p in prices],
        "close": prices,
        "volume": [50000 + i * 10 for i in range(100)],
        "symbol": "LEAKTEST.NS"
    })


def test_feature_creation(synthetic_ohlcv):
    df_feat = FeatureEngineer.create_features(synthetic_ohlcv)

    for col in FEATURE_COLUMNS:
        assert col in df_feat.columns, f"Feature {col} missing in engineered dataframe."
    assert TARGET_COLUMN in df_feat.columns


def test_no_future_data_leakage(synthetic_ohlcv):
    """Verify that features at index t do not know the price at index t+1."""
    df_feat = FeatureEngineer.create_features(synthetic_ohlcv)

    # Modify the future price at index 50
    t = 40
    future_index = 50

    # Ensure close_lag_1 at t equals close at t
    assert df_feat["close_lag_1"].iloc[t] == synthetic_ohlcv["close"].iloc[t]

    # Target at t should equal close at t+1
    assert df_feat[TARGET_COLUMN].iloc[t] == synthetic_ohlcv["close"].iloc[t + 1]

    # Verify that close_lag_1 at t does not change if future close at t+1 changes
    altered_ohlcv = synthetic_ohlcv.copy()
    altered_ohlcv.loc[t + 1, "close"] = 99999.0
    altered_feat = FeatureEngineer.create_features(altered_ohlcv)

    assert altered_feat["close_lag_1"].iloc[t] == df_feat["close_lag_1"].iloc[t]
    assert altered_feat["sma_5"].iloc[t] == df_feat["sma_5"].iloc[t]


def test_get_modeling_data_shapes(synthetic_ohlcv):
    df_feat = FeatureEngineer.create_features(synthetic_ohlcv)
    X, y, latest_features = FeatureEngineer.get_modeling_data(df_feat)

    assert len(X) == len(y)
    assert len(latest_features) == 1
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert list(latest_features.columns) == FEATURE_COLUMNS
