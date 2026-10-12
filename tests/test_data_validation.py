"""Unit tests for data validation module."""

import pytest
import pandas as pd
import numpy as np

from src.data_validation import DataValidator


@pytest.fixture
def valid_stock_dataframe():
    """Deterministic synthetic fixture for valid stock dataframe."""
    dates = pd.date_range("2025-01-01", periods=50, freq="B")
    return pd.DataFrame({
        "Date": dates,
        "Open": [100.0 + i for i in range(50)],
        "High": [105.0 + i for i in range(50)],
        "Low": [95.0 + i for i in range(50)],
        "Close": [102.0 + i for i in range(50)],
        "Volume": [100000 + i * 100 for i in range(50)]
    })


def test_normalize_columns(valid_stock_dataframe):
    norm = DataValidator.normalize_columns(valid_stock_dataframe)
    assert "date" in norm.columns
    assert "open" in norm.columns
    assert "high" in norm.columns
    assert "low" in norm.columns
    assert "close" in norm.columns
    assert "volume" in norm.columns


def test_validate_schema_success(valid_stock_dataframe):
    is_valid, errors = DataValidator.validate_schema(valid_stock_dataframe)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_schema_missing_column():
    invalid_df = pd.DataFrame({"Date": ["2025-01-01"], "Open": [100.0]})
    is_valid, errors = DataValidator.validate_schema(invalid_df)
    assert is_valid is False
    assert any("Missing required columns" in err for err in errors)


def test_audit_quality_valid(valid_stock_dataframe):
    report = DataValidator.audit_quality(valid_stock_dataframe, symbol="TEST.NS")
    assert report["is_valid"] is True
    assert report["total_records"] == 50
    assert report["valid_records"] == 50
    assert report["duplicate_dates"] == 0
    assert report["price_invariants_broken"] == 0


def test_audit_quality_with_anomalies():
    bad_df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-01", "2025-01-03"],  # Duplicate date
        "open": [100.0, -50.0, 102.0],  # Negative price
        "high": [90.0, 60.0, 110.0],    # High < Low on row 0
        "low": [95.0, 40.0, 100.0],
        "close": [92.0, 55.0, 105.0],
        "volume": [-100, 5000, 6000]    # Negative volume
    })
    report = DataValidator.audit_quality(bad_df, symbol="BAD.NS")
    assert report["duplicate_dates"] > 0
    assert report["price_invariants_broken"] > 0
    assert report["negative_volume_count"] > 0
