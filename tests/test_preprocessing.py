"""Unit tests for data preprocessing and cleaning module."""

import pytest
import pandas as pd
import numpy as np

from src.preprocessing import DataPreprocessor
from src.data_validation import DataValidationError


def test_clean_stock_data_sorting_and_deduplication():
    # Intentionally unsorted with duplicate date and invalid price
    raw_df = pd.DataFrame({
        "Date": ["2025-01-05", "2025-01-02", "2025-01-02", "2025-01-03", "2025-01-04"],
        "Open": [105.0, 101.0, 102.0, 103.0, -10.0],  # row with -10.0 should be dropped
        "High": [110.0, 106.0, 107.0, 108.0, 15.0],
        "Low": [100.0, 98.0, 99.0, 101.0, 5.0],
        "Close": [107.0, 103.0, 104.0, 105.0, 12.0],
        "Volume": [1000, 2000, 2500, 3000, 500]
    })

    cleaned = DataPreprocessor.clean_stock_data(raw_df, symbol="TEST.NS")

    # 1. Monotonic increasing dates
    assert cleaned["date"].is_monotonic_increasing

    # 2. Duplicate 2025-01-02 resolved to 1 entry
    assert (cleaned["date_str"] == "2025-01-02").sum() == 1

    # 3. Invalid negative price row was removed
    assert (cleaned["open"] <= 0).sum() == 0

    # 4. Symbol attached
    assert all(cleaned["symbol"] == "TEST.NS")


def test_clean_stock_data_empty_raises_error():
    empty_df = pd.DataFrame()
    with pytest.raises(DataValidationError):
        DataPreprocessor.clean_stock_data(empty_df, symbol="EMPTY.NS")
