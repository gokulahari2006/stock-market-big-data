"""Model evaluation module calculating statistical performance metrics:
MAE, RMSE, MAPE, and Directional Accuracy, with residual diagnostic helpers.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class ModelEvaluator:
    """Evaluates time-series regression models on out-of-sample test datasets."""

    @classmethod
    def calculate_metrics(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prev_close: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute comprehensive out-of-sample evaluation metrics.

        Args:
            y_true: Actual next-day closing prices.
            y_pred: Model predicted next-day closing prices.
            prev_close: Closing price at time t (used to evaluate directional accuracy).

        Returns:
            Dictionary with MAE, RMSE, MAPE (%), and Directional Accuracy (%).
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(y_true) == 0 or len(y_pred) == 0:
            return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "directional_accuracy": 0.0}

        # 1. Mean Absolute Error (MAE)
        mae = float(np.mean(np.abs(y_true - y_pred)))

        # 2. Root Mean Squared Error (RMSE)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        # 3. Mean Absolute Percentage Error (MAPE)
        # Avoid division by zero by filtering out near-zero true values
        mask = np.abs(y_true) > 1e-6
        if np.any(mask):
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
        else:
            mape = 0.0

        # 4. Directional Accuracy (Does predicted direction match actual direction?)
        dir_acc = 0.0
        if prev_close is not None and len(prev_close) == len(y_true):
            prev_close = np.asarray(prev_close, dtype=float)
            actual_direction = np.sign(y_true - prev_close)
            pred_direction = np.sign(y_pred - prev_close)

            # Match is when both are positive, both are negative, or both are zero
            matches = (actual_direction == pred_direction)
            dir_acc = float(np.mean(matches) * 100.0)
        else:
            # Fallback directional calculation comparing successive predicted points
            if len(y_true) > 1:
                actual_diff = np.sign(np.diff(y_true))
                pred_diff = np.sign(np.diff(y_pred))
                dir_acc = float(np.mean(actual_diff == pred_diff) * 100.0)

        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4),
            "directional_accuracy": round(dir_acc, 2)
        }

    @classmethod
    def analyze_residuals(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Compute residual statistics and diagnostic distributions."""
        residuals = np.asarray(y_true) - np.asarray(y_pred)
        return {
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "max_error": float(np.max(np.abs(residuals))),
            "residuals": residuals.tolist()
        }
