"""Financial analytics engine calculating core technical indicators,
moving averages, volatility, momentum metrics, and statistical summaries.
All calculations use only past information at each observation point.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.config import (
    SMA_WINDOWS,
    EMA_WINDOWS,
    VOLATILITY_WINDOW,
    RSI_WINDOW,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    BOLLINGER_WINDOW,
    BOLLINGER_STD_DEV
)
from src.utils import logger


class FinancialAnalytics:
    """Calculates quantitative financial metrics and technical indicators from OHLCV data."""

    @classmethod
    def calculate_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Compute complete suite of technical indicators on a chronologically sorted dataframe."""
        df = df.copy()

        # Ensure sorted by date
        if not df["date"].is_monotonic_increasing:
            df = df.sort_values(by="date", ascending=True).reset_index(drop=True)

        close = df["close"]

        # 1. Daily percentage returns
        df["daily_return_pct"] = close.pct_change() * 100.0
        df["daily_return_ratio"] = close.pct_change()

        # 2. Simple Moving Averages (SMA)
        for w in SMA_WINDOWS:
            col_name = f"sma_{w}"
            df[col_name] = close.rolling(window=w, min_periods=w).mean()

        # 3. Exponential Moving Averages (EMA)
        for w in EMA_WINDOWS:
            col_name = f"ema_{w}"
            df[col_name] = close.ewm(span=w, adjust=False).mean()

        # 4. Rolling Volatility (20-day standard deviation of daily returns)
        # Annualized volatility = daily_std * sqrt(252 trading days)
        daily_std = df["daily_return_pct"].rolling(window=VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).std()
        df[f"volatility_{VOLATILITY_WINDOW}"] = daily_std
        df[f"annualized_volatility_{VOLATILITY_WINDOW}"] = daily_std * np.sqrt(252)

        # 5. Trading Volume Analytics
        if "volume" in df.columns:
            df["volume_sma_20"] = df["volume"].rolling(window=20, min_periods=5).mean()
            df["volume_change_pct"] = df["volume"].pct_change() * 100.0

        # 6. Relative Strength Index (RSI - 14 period Wilder smoothed)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        # Exponential moving average of gains and losses
        avg_gain = gain.ewm(alpha=1.0 / RSI_WINDOW, min_periods=RSI_WINDOW, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / RSI_WINDOW, min_periods=RSI_WINDOW, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f"rsi_{RSI_WINDOW}"] = 100.0 - (100.0 / (1.0 + rs))

        # 7. Moving Average Convergence Divergence (MACD)
        ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # 8. Bollinger Bands (20-period SMA +/- 2 std dev)
        bb_mid = close.rolling(window=BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).mean()
        bb_std = close.rolling(window=BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).std()
        df["bollinger_mid"] = bb_mid
        df["bollinger_upper"] = bb_mid + (BOLLINGER_STD_DEV * bb_std)
        df["bollinger_lower"] = bb_mid - (BOLLINGER_STD_DEV * bb_std)

        # 9. Rolling extremes
        df["period_high_20"] = df["high"].rolling(window=20, min_periods=1).max()
        df["period_low_20"] = df["low"].rolling(window=20, min_periods=1).min()

        return df

    @classmethod
    def get_market_summary(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute key summary statistics for the overall selected period."""
        if df.empty:
            return {}

        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest

        latest_close = float(latest["close"])
        prev_close = float(previous["close"])
        abs_change = latest_close - prev_close
        pct_change = (abs_change / prev_close * 100.0) if prev_close != 0 else 0.0

        period_high = float(df["high"].max())
        period_low = float(df["low"].min())
        first_close = float(df.iloc[0]["close"])
        period_return_pct = ((latest_close - first_close) / first_close * 100.0) if first_close != 0 else 0.0

        avg_volume = float(df["volume"].mean()) if "volume" in df.columns else 0.0
        tot_volume = int(df["volume"].sum()) if "volume" in df.columns else 0

        # Volatility
        returns = df["close"].pct_change().dropna() * 100.0
        hist_volatility = float(returns.std()) if not returns.empty else 0.0
        annualized_vol = hist_volatility * np.sqrt(252)

        return {
            "latest_date": str(latest["date"])[:10],
            "latest_close": latest_close,
            "previous_close": prev_close,
            "absolute_change": abs_change,
            "percentage_change": pct_change,
            "period_high": period_high,
            "period_low": period_low,
            "highest_high": float(df["high"].max()),
            "lowest_low": float(df["low"].min()),
            "period_return_pct": period_return_pct,
            "avg_volume": avg_volume,
            "total_volume": tot_volume,
            "historical_daily_volatility_pct": hist_volatility,
            "annualized_volatility_pct": annualized_vol,
            "total_trading_days": len(df)
        }
