"""Stock market data collection module with yfinance integration,
local caching, offline fallback to sample datasets, and synthetic dataset generation.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    SAMPLE_DATA_DIR,
    SUPPORTED_TICKERS,
    DEFAULT_TICKER
)
from src.database import db
from src.utils import logger, safe_date_str

# Try importing yfinance safely
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False


class StockDataCollector:
    """Manages historical stock market data collection, caching, and fallback sample handling."""

    def __init__(self, raw_dir: Path = RAW_DATA_DIR, sample_dir: Path = SAMPLE_DATA_DIR):
        self.raw_dir = raw_dir
        self.sample_dir = sample_dir

    def fetch_stock_data(
        self,
        symbol: str = DEFAULT_TICKER,
        period: str = "1y",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> Tuple[pd.DataFrame, str, bool]:
        """Fetch historical stock data.

        Returns:
            Tuple of (DataFrame, source_label, is_sample_data)
        """
        symbol = symbol.strip().upper()
        clean_symbol = symbol.replace(".", "_")
        raw_cache_path = self.raw_dir / f"{clean_symbol}_raw.csv"

        # Check local cache first if not force_refresh and no custom date range
        if not force_refresh and raw_cache_path.exists() and period != "custom":
            try:
                cached_df = pd.read_csv(raw_cache_path)
                if not cached_df.empty and len(cached_df) > 10:
                    cached_df["date"] = pd.to_datetime(cached_df["date"])
                    logger.info(f"Loaded {len(cached_df)} records for {symbol} from local cache.")
                    db.log_ingestion(symbol, len(cached_df), "LOCAL_CACHE", "SUCCESS")
                    return cached_df, "Local Cache (Offline)", False
            except Exception as e:
                logger.warning(f"Failed reading cache for {symbol}: {e}")

        # Attempt live download via yfinance
        if YFINANCE_AVAILABLE:
            try:
                logger.info(f"Attempting live download for {symbol} (period={period})...")
                ticker_obj = yf.Ticker(symbol)

                if period == "custom" and start_date and end_date:
                    df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=False)
                else:
                    valid_period = period if period in ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"] else "1y"
                    df = ticker_obj.history(period=valid_period, auto_adjust=False)

                if df is not None and not df.empty and len(df) > 5:
                    df = df.reset_index()
                    df.columns = [str(c).lower() for c in df.columns]
                    # Save to raw cache
                    df.to_csv(raw_cache_path, index=False)
                    logger.info(f"Successfully downloaded {len(df)} records for {symbol} via yfinance.")
                    db.log_ingestion(symbol, len(df), "YFINANCE_LIVE", "SUCCESS")
                    return df, "Yahoo Finance Live Data (Delayed)", False
                else:
                    logger.warning(f"yfinance returned empty dataset for {symbol}.")
            except Exception as ex:
                logger.warning(f"Live data fetch error for {symbol}: {ex}")
                db.log_ingestion(symbol, 0, "YFINANCE_LIVE", f"FAILED: {str(ex)[:100]}")
        else:
            logger.warning("yfinance package is not installed; switching to sample mode.")

        # Fallback to sample dataset
        sample_path = self.sample_dir / f"{clean_symbol}_sample.csv"
        if sample_path.exists():
            sample_df = pd.read_csv(sample_path)
            sample_df["date"] = pd.to_datetime(sample_df["date"])
            logger.info(f"Loaded verified sample dataset for {symbol} ({len(sample_df)} records).")
            db.log_ingestion(symbol, len(sample_df), "SAMPLE_DATASET", "FALLBACK")
            return sample_df, "Verified Sample Dataset (Fallback)", True

        # If sample dataset does not exist, check any sample file or create a labeled sample
        generic_sample = next(self.sample_dir.glob("*_sample.csv"), None)
        if generic_sample and generic_sample.exists():
            generic_df = pd.read_csv(generic_sample)
            generic_df["date"] = pd.to_datetime(generic_df["date"])
            logger.info(f"Loaded generic sample dataset from {generic_sample.name}.")
            db.log_ingestion(symbol, len(generic_df), "GENERIC_SAMPLE", "FALLBACK")
            return generic_df, f"Generic Sample Dataset ({generic_sample.stem})", True

        # Last resort: generate clearly labeled sample data
        logger.info(f"Generating reproducible demonstration sample dataset for {symbol}.")
        demo_df = self.generate_sample_dataset(symbol=symbol, days=365)
        demo_df.to_csv(sample_path, index=False)
        db.log_ingestion(symbol, len(demo_df), "SYNTHETIC_GENERATOR", "FALLBACK")
        return demo_df, "Generated Sample Dataset (Clearly Labeled Demo)", True

    @classmethod
    def generate_sample_dataset(
        cls,
        symbol: str = "RELIANCE.NS",
        days: int = 365,
        base_price: float = 2500.0,
        volatility: float = 0.018,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """Generate a reproducible, mathematically realistic historical stock series for offline demonstration.

        Uses Geometric Brownian Motion (GBM) with realistic daily volume and OHLC spreads.
        Clearly labeled as sample demonstration data.
        """
        np.random.seed(random_seed)
        end_dt = datetime.now()
        # Generate business days
        date_range = pd.date_range(end=end_dt, periods=days, freq="B")

        # Geometric Brownian Motion simulation: S_t = S_0 * exp(cumsum((mu - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z))
        dt = 1.0 / 252.0
        mu = 0.10  # 10% annual expected return
        sigma = volatility * np.sqrt(252.0)

        daily_shocks = np.random.normal(0, 1, size=len(date_range))
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * daily_shocks
        price_multipliers = np.exp(np.cumsum(log_returns))
        close_prices = base_price * price_multipliers

        records = []
        for d, close in zip(date_range, close_prices):
            intraday_vol = np.random.uniform(0.005, 0.02)
            open_p = close * (1.0 + np.random.normal(0, 0.005))
            high_p = max(open_p, close) * (1.0 + intraday_vol)
            low_p = min(open_p, close) * (1.0 - intraday_vol)
            vol = int(np.random.lognormal(mean=14.5, sigma=0.5))  # Realistic volume ~2M shares

            records.append({
                "date": d.strftime("%Y-%m-%d"),
                "open": round(float(open_p), 2),
                "high": round(float(high_p), 2),
                "low": round(float(low_p), 2),
                "close": round(float(close), 2),
                "adj_close": round(float(close), 2),
                "volume": vol,
                "symbol": symbol
            })

        return pd.DataFrame(records)


collector = StockDataCollector()
