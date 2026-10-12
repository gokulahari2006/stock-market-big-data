"""Unit tests for Spark processor and fallback analytics engine."""

import pytest
import pandas as pd
import numpy as np

from src.spark_processor import SparkProcessor


@pytest.fixture
def multi_stock_df():
    dates = pd.date_range("2025-01-01", periods=30, freq="B")
    records = []
    for sym in ["RELIANCE.NS", "TCS.NS"]:
        for i, d in enumerate(dates):
            base = 2000.0 if sym == "RELIANCE.NS" else 3500.0
            price = base + i * 10.0
            records.append({
                "date": d.strftime("%Y-%m-%d"),
                "open": price,
                "high": price + 5.0,
                "low": price - 5.0,
                "close": price,
                "volume": 100000 + i * 100,
                "symbol": sym
            })
    return pd.DataFrame(records)


def test_spark_processor_initialization():
    engine = SparkProcessor()
    assert hasattr(engine, "spark_available")
    assert hasattr(engine, "run_spark_pipeline")


def test_spark_or_fallback_pipeline(multi_stock_df):
    engine = SparkProcessor()
    result = engine.run_spark_pipeline(multi_stock_df)

    assert result["status"] == "SUCCESS"
    assert result["input_records"] == len(multi_stock_df)
    assert result["output_records"] > 0
    assert result["symbols_processed"] == 2
    assert "stock_aggregations" in result
    assert "date_aggregations" in result
    assert len(result["stock_aggregations"]) == 2


def test_empty_input_handling():
    engine = SparkProcessor()
    result = engine.run_spark_pipeline(pd.DataFrame())
    assert result["status"] == "EMPTY_INPUT"
