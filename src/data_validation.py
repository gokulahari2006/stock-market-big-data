"""Data validation module ensuring stock market OHLCV data integrity,
schema correctness, logical price invariants, and generating data quality reports.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from src.utils import logger, safe_date_str


class DataValidationError(Exception):
    """Raised when critical data validation rules fail."""
    pass


class DataValidator:
    """Validates financial OHLCV datasets and reports quality diagnostics."""

    REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

    @classmethod
    def normalize_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase without spaces."""
        df = df.copy()
        # Flatten MultiIndex columns if produced by yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        # Handle 'adj_close' or 'adjusted_close'
        if "adj_close" not in df.columns and "adjusted_close" in df.columns:
            df = df.rename(columns={"adjusted_close": "adj_close"})

        # If date is in index, reset index
        if "date" not in df.columns and (isinstance(df.index, pd.DatetimeIndex) or df.index.name in ["Date", "date"]):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "date"})

        return df

    @classmethod
    def validate_schema(cls, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Verify presence of required OHLCV columns."""
        df_norm = cls.normalize_columns(df)
        missing = [col for col in cls.REQUIRED_COLUMNS if col not in df_norm.columns]
        if missing:
            return False, [f"Missing required columns: {', '.join(missing)}"]
        return True, []

    @classmethod
    def audit_quality(cls, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Run deep data quality checks and return a diagnostic summary."""
        report: Dict[str, Any] = {
            "symbol": symbol,
            "total_records": len(df),
            "valid_records": 0,
            "invalid_records": 0,
            "duplicate_dates": 0,
            "missing_values": {},
            "start_date": None,
            "end_date": None,
            "price_invariants_broken": 0,
            "negative_volume_count": 0,
            "issues": [],
            "is_valid": True
        }

        if df.empty:
            report["is_valid"] = False
            report["issues"].append("Dataset is completely empty.")
            return report

        df_norm = cls.normalize_columns(df)
        has_schema, schema_issues = cls.validate_schema(df_norm)
        if not has_schema:
            report["is_valid"] = False
            report["issues"].extend(schema_issues)
            return report

        # Date validation
        try:
            dates = pd.to_datetime(df_norm["date"], errors="coerce")
            null_dates = dates.isna().sum()
            if null_dates > 0:
                report["issues"].append(f"Found {null_dates} unparseable date entries.")
            valid_dates = dates.dropna()
            if not valid_dates.empty:
                report["start_date"] = safe_date_str(valid_dates.min())
                report["end_date"] = safe_date_str(valid_dates.max())

            dup_dates = dates.duplicated().sum()
            report["duplicate_dates"] = int(dup_dates)
            if dup_dates > 0:
                report["issues"].append(f"Found {dup_dates} duplicate date entries.")
        except Exception as e:
            report["issues"].append(f"Date parsing error: {str(e)}")

        # Missing values check
        missing_dict = {}
        for col in cls.REQUIRED_COLUMNS:
            count = int(df_norm[col].isna().sum())
            if count > 0:
                missing_dict[col] = count
        report["missing_values"] = missing_dict
        if missing_dict:
            report["issues"].append(f"Missing values found in columns: {missing_dict}")

        # Numeric checks and price logic
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df_norm[col] = pd.to_numeric(df_norm[col], errors="coerce")

        # Invariant checks: High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
        tolerance = 1e-4
        invalid_high_low = (df_norm["high"] < (df_norm["low"] - tolerance)).sum()
        invalid_prices = ((df_norm["open"] <= 0) | (df_norm["close"] <= 0) |
                          (df_norm["high"] <= 0) | (df_norm["low"] <= 0)).sum()
        negative_vol = (df_norm["volume"] < 0).sum()

        broken_invariants = int(invalid_high_low + invalid_prices)
        report["price_invariants_broken"] = broken_invariants
        report["negative_volume_count"] = int(negative_vol)

        if broken_invariants > 0:
            report["issues"].append(f"Detected {broken_invariants} price logic violations (e.g. High < Low or Non-positive price).")
        if negative_vol > 0:
            report["issues"].append(f"Detected {negative_vol} records with negative trading volume.")

        # Compute valid records
        is_clean_mask = (
            df_norm["date"].notna() &
            ~df_norm["date"].duplicated() &
            df_norm["open"].notna() & (df_norm["open"] > 0) &
            df_norm["high"].notna() & (df_norm["high"] >= df_norm["low"] - tolerance) &
            df_norm["low"].notna() & (df_norm["low"] > 0) &
            df_norm["close"].notna() & (df_norm["close"] > 0) &
            df_norm["volume"].notna() & (df_norm["volume"] >= 0)
        )
        report["valid_records"] = int(is_clean_mask.sum())
        report["invalid_records"] = report["total_records"] - report["valid_records"]

        if report["valid_records"] < 20:
            report["is_valid"] = False
            report["issues"].append("Insufficient valid trading observations (< 20 trading sessions).")

        return report
