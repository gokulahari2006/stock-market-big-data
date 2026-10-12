"""Unit tests for SQLite database operations and audit logging."""

import pytest
from pathlib import Path
import tempfile

from src.database import DatabaseManager


@pytest.fixture
def temp_db(tmp_path):
    temp_path = tmp_path / "test_stock.db"
    manager = DatabaseManager(db_path=temp_path)
    return manager


def test_log_ingestion(temp_db):
    row_id = temp_db.log_ingestion("TEST.NS", 150, "UNIT_TEST", "SUCCESS")
    assert row_id > 0

    logs = temp_db.get_recent_ingestions(limit=5)
    assert len(logs) == 1
    assert logs[0]["symbol"] == "TEST.NS"
    assert logs[0]["record_count"] == 150


def test_save_and_get_evaluations(temp_db):
    row_id = temp_db.save_model_evaluation(
        symbol="RELIANCE.NS",
        model_name="Random Forest",
        train_size=100,
        test_size=25,
        mae=12.5,
        rmse=18.2,
        mape=0.65,
        directional_accuracy=64.0
    )
    assert row_id > 0

    evals = temp_db.get_evaluations(symbol="RELIANCE.NS")
    assert len(evals) == 1
    assert evals[0]["model_name"] == "Random Forest"
    assert evals[0]["mae"] == 12.5


def test_save_and_get_predictions(temp_db):
    row_id = temp_db.save_prediction(
        symbol="TCS.NS",
        model_name="Linear Regression",
        last_known_date="2026-10-10",
        last_close=3850.0,
        predicted_next_close=3875.0,
        predicted_change_pct=0.65
    )
    assert row_id > 0

    preds = temp_db.get_predictions(symbol="TCS.NS")
    assert len(preds) == 1
    assert preds[0]["predicted_next_close"] == 3875.0
