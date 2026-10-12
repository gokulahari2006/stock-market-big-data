"""Machine learning prediction module supporting Linear Regression,
Random Forest Regression, Gradient Boosting, and a Persistence Baseline.
Enforces chronological train/test splitting and safe artifact persistence with joblib.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from src.config import (
    TRAIN_TEST_SPLIT_RATIO,
    MODEL_HYPERPARAMETERS,
    MODELS_DIR,
    FEATURE_COLUMNS,
    PROJECT_DISCLAIMER
)
from src.database import db
from src.model_evaluation import ModelEvaluator
from src.utils import logger


class StockPredictor:
    """Manages model training, chronological evaluation, next-session inference, and model persistence."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.scaler = StandardScaler()
        self.trained_models: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None

    def chronological_train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        split_ratio: float = TRAIN_TEST_SPLIT_RATIO
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split time-series data chronologically without shuffling to preserve causality."""
        n_samples = len(X)
        split_idx = int(n_samples * split_ratio)

        X_train = X.iloc[:split_idx].copy()
        X_test = X.iloc[split_idx:].copy()
        y_train = y.iloc[:split_idx].copy()
        y_test = y.iloc[split_idx:].copy()

        logger.info(f"Chronological split: Train = {len(X_train)} sessions, Test = {len(X_test)} sessions.")
        return X_train, X_test, y_train, y_test

    def train_and_evaluate_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        symbol: str = "RELIANCE.NS"
    ) -> Dict[str, Any]:
        """Train baseline and machine learning models on chronological train set and evaluate on test set."""
        X_train, X_test, y_train, y_test = self.chronological_train_test_split(X, y)

        # Baseline: Persistence Model (Predict next close = today's close, which is close_lag_1)
        prev_close_test = X_test["close_lag_1"].values
        persistence_preds = prev_close_test.copy()
        persistence_metrics = ModelEvaluator.calculate_metrics(
            y_true=y_test.values,
            y_pred=persistence_preds,
            prev_close=prev_close_test
        )

        results = {
            "Persistence Baseline": {
                "metrics": persistence_metrics,
                "predictions": persistence_preds,
                "model_obj": None
            }
        }

        # Fit Scaler strictly on X_train to prevent data leakage
        X_train_scaled = self.scaler.fit_transform(X_train[FEATURE_COLUMNS])
        X_test_scaled = self.scaler.transform(X_test[FEATURE_COLUMNS])

        # 1. Linear Regression
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        lr_preds = lr.predict(X_test_scaled)
        lr_metrics = ModelEvaluator.calculate_metrics(
            y_true=y_test.values,
            y_pred=lr_preds,
            prev_close=prev_close_test
        )
        results["Linear Regression"] = {
            "metrics": lr_metrics,
            "predictions": lr_preds,
            "model_obj": lr
        }

        # 2. Random Forest Regressor
        rf_params = MODEL_HYPERPARAMETERS["RandomForest"]
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_train[FEATURE_COLUMNS], y_train)  # Tree models do not strictly require scaling
        rf_preds = rf.predict(X_test[FEATURE_COLUMNS])
        rf_metrics = ModelEvaluator.calculate_metrics(
            y_true=y_test.values,
            y_pred=rf_preds,
            prev_close=prev_close_test
        )
        results["Random Forest"] = {
            "metrics": rf_metrics,
            "predictions": rf_preds,
            "model_obj": rf
        }

        # 3. Gradient Boosting Regressor
        gb_params = MODEL_HYPERPARAMETERS["GradientBoosting"]
        gb = GradientBoostingRegressor(**gb_params)
        gb.fit(X_train[FEATURE_COLUMNS], y_train)
        gb_preds = gb.predict(X_test[FEATURE_COLUMNS])
        gb_metrics = ModelEvaluator.calculate_metrics(
            y_true=y_test.values,
            y_pred=gb_preds,
            prev_close=prev_close_test
        )
        results["Gradient Boosting"] = {
            "metrics": gb_metrics,
            "predictions": gb_preds,
            "model_obj": gb
        }

        self.trained_models = {
            "Linear Regression": lr,
            "Random Forest": rf,
            "Gradient Boosting": gb
        }
        self.evaluation_results = results

        # Determine best ML model based on lowest test RMSE
        best_name = "Random Forest"
        lowest_rmse = float("inf")
        for m_name in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
            rmse_val = results[m_name]["metrics"]["rmse"]
            if rmse_val < lowest_rmse:
                lowest_rmse = rmse_val
                best_name = m_name
        self.best_model_name = best_name

        # Save evaluations to database
        for m_name, data in results.items():
            m = data["metrics"]
            db.save_model_evaluation(
                symbol=symbol,
                model_name=m_name,
                train_size=len(X_train),
                test_size=len(X_test),
                mae=m["mae"],
                rmse=m["rmse"],
                mape=m["mape"],
                directional_accuracy=m["directional_accuracy"]
            )

        return {
            "results": results,
            "best_model": best_name,
            "y_test": y_test.values,
            "dates_test": X_test.index if hasattr(X_test, "index") else None,
            "prev_close_test": prev_close_test,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }

    def predict_next_session(
        self,
        latest_features: pd.DataFrame,
        model_name: str,
        symbol: str = "RELIANCE.NS"
    ) -> Dict[str, Any]:
        """Generate prediction for the upcoming trading session using the latest available features."""
        if model_name not in self.trained_models and model_name != "Persistence Baseline":
            raise ValueError(f"Model '{model_name}' has not been trained yet.")

        latest_close = float(latest_features["close_lag_1"].iloc[0])

        if model_name == "Persistence Baseline":
            predicted_close = latest_close
        elif model_name == "Linear Regression":
            feat_scaled = self.scaler.transform(latest_features[FEATURE_COLUMNS])
            predicted_close = float(self.trained_models[model_name].predict(feat_scaled)[0])
        else:
            predicted_close = float(self.trained_models[model_name].predict(latest_features[FEATURE_COLUMNS])[0])

        abs_diff = predicted_close - latest_close
        pct_diff = (abs_diff / latest_close * 100.0) if latest_close != 0 else 0.0

        # Save prediction history to SQLite
        db.save_prediction(
            symbol=symbol,
            model_name=model_name,
            last_known_date=datetime.now().strftime("%Y-%m-%d"),
            last_close=latest_close,
            predicted_next_close=predicted_close,
            predicted_change_pct=pct_diff
        )

        return {
            "symbol": symbol,
            "model_name": model_name,
            "latest_close": round(latest_close, 2),
            "predicted_next_close": round(predicted_close, 2),
            "absolute_change": round(abs_diff, 2),
            "percentage_change": round(pct_diff, 2),
            "prediction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "disclaimer": PROJECT_DISCLAIMER
        }

    def save_model(self, symbol: str, model_name: str) -> Path:
        """Serialize trained model artifact safely using joblib."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} is not trained.")

        self.models_dir.mkdir(parents=True, exist_ok=True)
        clean_symbol = symbol.replace(".", "_")
        clean_model = model_name.replace(" ", "_").lower()
        save_path = self.models_dir / f"{clean_symbol}_{clean_model}.joblib"

        payload = {
            "symbol": symbol,
            "model_name": model_name,
            "model_obj": self.trained_models[model_name],
            "scaler": self.scaler,
            "features": FEATURE_COLUMNS,
            "saved_at": datetime.now().isoformat()
        }
        joblib.dump(payload, save_path)
        logger.info(f"Model artifact saved to {save_path}")
        return save_path

    def load_model(self, save_path: Path) -> Dict[str, Any]:
        """Safely load and validate serialized model artifact."""
        if not save_path.exists():
            raise FileNotFoundError(f"Model file not found: {save_path}")

        payload = joblib.load(save_path)
        if not isinstance(payload, dict) or "model_obj" not in payload or "features" not in payload:
            raise ValueError("Corrupted or unauthorized model file format.")

        return payload
