"""Unit tests for machine learning prediction models and evaluation metrics."""

import pytest
import numpy as np
import pandas as pd

from src.prediction_model import StockPredictor
from src.model_evaluation import ModelEvaluator
from src.feature_engineering import FeatureEngineer


@pytest.fixture
def modeling_dataset():
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    prices = [1000.0 + i * 5.0 + (i % 3) * 2.0 for i in range(120)]
    df = pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": [p + 5.0 for p in prices],
        "low": [p - 5.0 for p in prices],
        "close": prices,
        "volume": [100000] * 120,
        "symbol": "PREDTEST.NS"
    })
    df_feat = FeatureEngineer.create_features(df)
    return FeatureEngineer.get_modeling_data(df_feat)


def test_chronological_split(modeling_dataset):
    X, y, _ = modeling_dataset
    predictor = StockPredictor()
    X_train, X_test, y_train, y_test = predictor.chronological_train_test_split(X, y, split_ratio=0.8)

    assert len(X_train) + len(X_test) == len(X)
    assert len(X_train) == int(len(X) * 0.8)
    # Chronological integrity: indices of train must strictly precede test
    assert X_train.index.max() < X_test.index.min()


def test_train_and_evaluate_all(modeling_dataset):
    X, y, _ = modeling_dataset
    predictor = StockPredictor()
    results_dict = predictor.train_and_evaluate_all(X, y, symbol="PREDTEST.NS")

    res = results_dict["results"]
    assert "Persistence Baseline" in res
    assert "Linear Regression" in res
    assert "Random Forest" in res

    for model_name, data in res.items():
        metrics = data["metrics"]
        assert "mae" in metrics and metrics["mae"] >= 0.0
        assert "rmse" in metrics and metrics["rmse"] >= 0.0
        assert "directional_accuracy" in metrics
        assert 0.0 <= metrics["directional_accuracy"] <= 100.0


def test_next_session_prediction(modeling_dataset):
    X, y, latest_features = modeling_dataset
    predictor = StockPredictor()
    predictor.train_and_evaluate_all(X, y, symbol="PREDTEST.NS")

    forecast = predictor.predict_next_session(latest_features, "Random Forest", symbol="PREDTEST.NS")
    assert forecast["symbol"] == "PREDTEST.NS"
    assert "predicted_next_close" in forecast
    assert forecast["predicted_next_close"] > 0
    assert "absolute_change" in forecast
    assert "percentage_change" in forecast


def test_model_evaluator_metrics():
    y_true = np.array([100.0, 105.0, 102.0, 108.0])
    y_pred = np.array([99.0, 104.0, 103.0, 107.0])
    prev_close = np.array([98.0, 100.0, 105.0, 102.0])

    metrics = ModelEvaluator.calculate_metrics(y_true, y_pred, prev_close)
    assert metrics["mae"] == 1.0
    assert metrics["rmse"] == 1.0
    assert metrics["directional_accuracy"] == 100.0
