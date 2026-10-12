"""Feature engineering module for predictive modeling of next-trading-session closing price.
Enforces strict prevention of data leakage by ensuring that all features at row t
use strictly historical observations available on or before trading session t.
"""

from typing import Tuple, List
import pandas as pd
import numpy as np

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.financial_analytics import FinancialAnalytics
from src.utils import logger


class FeatureEngineeringError(Exception):
    """Raised when feature engineering encounters invalid or insufficient data."""
    pass


class FeatureEngineer:
    """Constructs lagged price/volume features and technical indicators without target leakage."""

    @classmethod
    def create_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all predictive features and target variable from clean OHLCV dataframe.

        Target:
            target_next_close = close at t+1 (shifted backward by 1 session)

        Features at row t:
            close_lag_1: Close at t
            close_lag_2: Close at t-1
            close_lag_3: Close at t-2
            close_lag_5: Close at t-4
            close_lag_10: Close at t-9
            close_lag_20: Close at t-19
            return_lag_1: Return between t and t-1
            return_lag_5: Return between t and t-5
            return_lag_10: Return between t and t-10
            volume_lag_1: Volume at t
            volume_lag_5: Volume at t-4
            sma_5, sma_10, sma_20: Moving averages up to t
            ema_12, ema_26: Exponential moving averages up to t
            volatility_20: 20-day rolling return volatility up to t
            rsi_14: 14-day RSI up to t
            macd, macd_signal: MACD indicators up to t
        """
        df = df.copy()

        # 1. Calculate technical indicators (all strictly causal up to row t)
        df = FinancialAnalytics.calculate_all_indicators(df)

        # 2. Lagged Close Prices
        df["close_lag_1"] = df["close"]
        df["close_lag_2"] = df["close"].shift(1)
        df["close_lag_3"] = df["close"].shift(2)
        df["close_lag_5"] = df["close"].shift(4)
        df["close_lag_10"] = df["close"].shift(9)
        df["close_lag_20"] = df["close"].shift(19)

        # 3. Lagged Returns
        df["return_lag_1"] = df["close"].pct_change(1) * 100.0
        df["return_lag_5"] = df["close"].pct_change(5) * 100.0
        df["return_lag_10"] = df["close"].pct_change(10) * 100.0

        # 4. Lagged Volume
        df["volume_lag_1"] = df["volume"]
        df["volume_lag_5"] = df["volume"].shift(4)

        # 5. Define Target: Close price of the next trading session (t + 1)
        df[TARGET_COLUMN] = df["close"].shift(-1)

        return df

    @classmethod
    def get_modeling_data(cls, df_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Extract training/testing feature matrix X, target y, and the latest inference row.

        Returns:
            Tuple of:
            - X: DataFrame of features for historical sessions with known next-day target.
            - y: Series of next-day closing prices (target).
            - latest_features: 1-row DataFrame containing features of the most recent session
                               used to forecast the upcoming unobserved session.
        """
        # Validate required columns
        for col in FEATURE_COLUMNS:
            if col not in df_features.columns:
                raise FeatureEngineeringError(f"Missing engineered feature: {col}")

        # The very last row is for forecasting tomorrow's unseen close
        latest_row = df_features.iloc[[-1]].copy()
        latest_features = latest_row[FEATURE_COLUMNS].copy()

        # Historical training data: only rows where both features AND target are known (not null)
        historical = df_features.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()

        if len(historical) < 30:
            raise FeatureEngineeringError(
                f"Insufficient records for model training: only {len(historical)} valid feature rows after warm-up."
            )

        X = historical[FEATURE_COLUMNS].copy()
        y = historical[TARGET_COLUMN].copy()

        return X, y, latest_features
