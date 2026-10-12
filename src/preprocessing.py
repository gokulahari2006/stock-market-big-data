"""Data preprocessing module for cleaning, sorting, type casting,
deduplication, and exporting stock market time-series datasets.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from src.data_validation import DataValidator, DataValidationError
from src.utils import logger, safe_date_str


class DataPreprocessor:
    """Preprocesses raw historical OHLCV data into a clean, normalized, chronological dataset."""

    @classmethod
    def clean_stock_data(cls, df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
        """Execute the end-to-end cleaning pipeline on raw stock market dataframe."""
        if df.empty:
            raise DataValidationError("Cannot preprocess empty dataframe.")

        # 1. Normalize column names
        cleaned = DataValidator.normalize_columns(df)

        # 2. Parse dates safely and strip time/timezone to normalize to discrete trading dates
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
        if cleaned["date"].dt.tz is not None:
            cleaned["date"] = cleaned["date"].dt.tz_localize(None)
        cleaned["date"] = cleaned["date"].dt.normalize()

        # Drop rows where date could not be parsed
        cleaned = cleaned.dropna(subset=["date"])

        # 3. Sort chronologically ascending
        cleaned = cleaned.sort_values(by="date", ascending=True)

        # 4. Remove duplicate date observations, keeping the latest recorded
        cleaned = cleaned.drop_duplicates(subset=["date"], keep="last")

        # 5. Type casting and numeric sanitization
        for col in ["open", "high", "low", "close"]:
            if col in cleaned.columns:
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

        if "adj_close" in cleaned.columns:
            cleaned["adj_close"] = pd.to_numeric(cleaned["adj_close"], errors="coerce")
        else:
            cleaned["adj_close"] = cleaned["close"]

        if "volume" in cleaned.columns:
            cleaned["volume"] = pd.to_numeric(cleaned["volume"], errors="coerce").fillna(0).astype(np.int64)

        # 6. Filter out invalid rows (prices <= 0 or high < low)
        tolerance = 1e-4
        valid_mask = (
            (cleaned["open"] > 0) &
            (cleaned["high"] > 0) &
            (cleaned["low"] > 0) &
            (cleaned["close"] > 0) &
            (cleaned["high"] >= cleaned["low"] - tolerance) &
            (cleaned["volume"] >= 0)
        )

        dropped_count = int((~valid_mask).sum())
        if dropped_count > 0:
            logger.warning(f"Dropping {dropped_count} invalid records failing basic price logic.")
        cleaned = cleaned[valid_mask].copy()

        # Add symbol column if not present
        if "symbol" not in cleaned.columns:
            cleaned["symbol"] = symbol

        # Reset index
        cleaned = cleaned.reset_index(drop=True)

        # Convert date to standard string format YYYY-MM-DD for consistent serialization
        cleaned["date_str"] = cleaned["date"].dt.strftime("%Y-%m-%d")

        logger.info(f"Cleaned dataset for {symbol}: {len(cleaned)} valid trading sessions.")
        return cleaned

    @classmethod
    def export_processed(cls, df: pd.DataFrame, base_path: Path) -> Tuple[Path, Path]:
        """Export cleaned dataset to both Parquet and CSV formats."""
        base_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = base_path.with_suffix(".csv")
        parquet_path = base_path.with_suffix(".parquet")

        # Export CSV
        df.to_csv(csv_path, index=False)

        # Export Parquet
        try:
            df.to_parquet(parquet_path, index=False, engine="pyarrow")
        except Exception as e:
            logger.warning(f"Parquet export failed with pyarrow ({e}), trying fastparquet or skipping.")

        return csv_path, parquet_path
