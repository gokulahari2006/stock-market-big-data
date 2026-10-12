"""Unit tests for financial analytics and technical indicators."""

import pytest
import pandas as pd
import numpy as np

from src.financial_analytics import FinancialAnalytics


@pytest.fixture
def sample_clean_df():
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    prices = [100.0 * (1.01 ** i) for i in range(60)]
    return pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": [p * 1.02 for p in prices],
        "low": [p * 0.98 for p in prices],
        "close": prices,
        "volume": [100000] * 60,
        "symbol": "MOCK.NS"
    })


def test_daily_return_calculations(sample_clean_df):
    df_res = FinancialAnalytics.calculate_all_indicators(sample_clean_df)

    assert "daily_return_pct" in df_res.columns
    # First row return should be NaN
    assert np.isnan(df_res["daily_return_pct"].iloc[0])
    # Next rows should have ~1.0% return
    assert pytest.approx(df_res["daily_return_pct"].iloc[1], 0.05) == 1.0


def test_moving_averages(sample_clean_df):
    df_res = FinancialAnalytics.calculate_all_indicators(sample_clean_df)

    assert "sma_5" in df_res.columns
    assert "sma_20" in df_res.columns
    assert "ema_12" in df_res.columns
    assert "ema_26" in df_res.columns

    # Check 5-day SMA value at index 4 (5th day)
    expected_sma5 = sample_clean_df["close"].iloc[:5].mean()
    assert pytest.approx(df_res["sma_5"].iloc[4], 1e-4) == expected_sma5


def test_bollinger_bands_and_rsi(sample_clean_df):
    df_res = FinancialAnalytics.calculate_all_indicators(sample_clean_df)

    assert "bollinger_upper" in df_res.columns
    assert "bollinger_lower" in df_res.columns
    assert "rsi_14" in df_res.columns

    # Bollinger Upper should be >= Bollinger Lower
    valid_bb = df_res.dropna(subset=["bollinger_upper", "bollinger_lower"])
    assert all(valid_bb["bollinger_upper"] >= valid_bb["bollinger_lower"])

    # RSI bounded between 0 and 100
    valid_rsi = df_res["rsi_14"].dropna()
    assert all((valid_rsi >= 0.0) & (valid_rsi <= 100.0))


def test_market_summary(sample_clean_df):
    summary = FinancialAnalytics.get_market_summary(sample_clean_df)
    assert "latest_close" in summary
    assert "period_high" in summary
    assert "period_low" in summary
    assert summary["latest_close"] == sample_clean_df["close"].iloc[-1]
    assert summary["period_high"] == sample_clean_df["high"].max()
