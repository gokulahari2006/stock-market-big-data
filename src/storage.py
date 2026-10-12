"""File storage management module for handling raw CSVs, processed Parquet files,
sample datasets, and serialized machine learning models.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import os

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_DATA_DIR,
    MODELS_DIR
)
from src.utils import logger


class StorageManager:
    """Provides organized file access, cataloging, and safe export operations."""

    def __init__(self):
        self.raw_dir = RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.sample_dir = SAMPLE_DATA_DIR
        self.models_dir = MODELS_DIR

    def get_raw_file_path(self, symbol: str) -> Path:
        """Get canonical raw data CSV path for a ticker."""
        clean_symbol = symbol.replace(".", "_")
        return self.raw_dir / f"{clean_symbol}_raw.csv"

    def get_processed_file_path(self, symbol: str, ext: str = "parquet") -> Path:
        """Get canonical processed data file path for a ticker."""
        clean_symbol = symbol.replace(".", "_")
        return self.processed_dir / f"{clean_symbol}_processed.{ext}"

    def get_sample_file_path(self, symbol: str) -> Path:
        """Get sample dataset path for a ticker."""
        clean_symbol = symbol.replace(".", "_")
        return self.sample_dir / f"{clean_symbol}_sample.csv"

    def get_model_path(self, symbol: str, model_name: str) -> Path:
        """Get serialized model artifact path."""
        clean_symbol = symbol.replace(".", "_")
        clean_model = model_name.replace(" ", "_").lower()
        return self.models_dir / f"{clean_symbol}_{clean_model}.joblib"

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets across raw, processed, and sample folders with metadata."""
        datasets = []
        directories = [
            ("Raw (CSV)", self.raw_dir, "*.csv"),
            ("Processed (Parquet)", self.processed_dir, "*.parquet"),
            ("Processed (CSV)", self.processed_dir, "*.csv"),
            ("Sample (CSV)", self.sample_dir, "*.csv"),
        ]

        for category, directory, pattern in directories:
            if not directory.exists():
                continue
            for file_path in directory.glob(pattern):
                stat = file_path.stat()
                datasets.append({
                    "name": file_path.name,
                    "category": category,
                    "path": str(file_path),
                    "size_kb": round(stat.st_size / 1024, 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "extension": file_path.suffix.lower()
                })

        return sorted(datasets, key=lambda x: x["modified"], reverse=True)

    def list_saved_models(self) -> List[Dict[str, Any]]:
        """List serialized model artifacts."""
        models = []
        if not self.models_dir.exists():
            return models

        for file_path in self.models_dir.glob("*.joblib"):
            stat = file_path.stat()
            models.append({
                "name": file_path.name,
                "path": str(file_path),
                "size_kb": round(stat.st_size / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
        return sorted(models, key=lambda x: x["modified"], reverse=True)


# Singleton instance
storage = StorageManager()
