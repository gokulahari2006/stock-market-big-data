"""Database management module using SQLite for persistent application metadata,
ingestion audit trails, model performance metrics, and prediction tracking.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator

from src.config import DB_PATH
from src.utils import logger


class DatabaseManager:
    """Manages SQLite operations safely with automatic schema migrations and parameterized queries."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Create and return a configured SQLite connection with foreign keys enabled, ensuring safe closure."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Create required tables and indexes if they do not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Ingestion logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_ingestion_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    fetch_time TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL
                )
            """)

            # 2. Model evaluations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    train_size INTEGER NOT NULL,
                    test_size INTEGER NOT NULL,
                    mae REAL NOT NULL,
                    rmse REAL NOT NULL,
                    mape REAL,
                    directional_accuracy REAL NOT NULL,
                    evaluated_at TEXT NOT NULL
                )
            """)

            # 3. Prediction history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    last_known_date TEXT NOT NULL,
                    last_close REAL NOT NULL,
                    predicted_next_close REAL NOT NULL,
                    predicted_change_pct REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # 4. Data quality summaries
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    total_rows INTEGER NOT NULL,
                    valid_rows INTEGER NOT NULL,
                    missing_values INTEGER NOT NULL,
                    duplicate_rows INTEGER NOT NULL,
                    report_json TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            conn.commit()
            logger.info(f"Database initialized successfully at {self.db_path}")

    def log_ingestion(self, symbol: str, record_count: int, source: str, status: str = "SUCCESS") -> int:
        """Record a data collection audit entry."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO data_ingestion_log (symbol, fetch_time, record_count, source, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (symbol, datetime.now().isoformat(), record_count, source, status)
            )
            conn.commit()
            return cursor.lastrowid

    def save_model_evaluation(
        self,
        symbol: str,
        model_name: str,
        train_size: int,
        test_size: int,
        mae: float,
        rmse: float,
        mape: Optional[float],
        directional_accuracy: float
    ) -> int:
        """Persist offline model evaluation metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO model_evaluations (
                    symbol, model_name, train_size, test_size, mae, rmse, mape, directional_accuracy, evaluated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (symbol, model_name, train_size, test_size, mae, rmse, mape, directional_accuracy, datetime.now().isoformat())
            )
            conn.commit()
            return cursor.lastrowid

    def get_evaluations(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve model evaluation records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT * FROM model_evaluations WHERE symbol = ? ORDER BY evaluated_at DESC LIMIT ?",
                    (symbol, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM model_evaluations ORDER BY evaluated_at DESC LIMIT ?",
                    (limit,)
                )
            return [dict(row) for row in cursor.fetchall()]

    def save_prediction(
        self,
        symbol: str,
        model_name: str,
        last_known_date: str,
        last_close: float,
        predicted_next_close: float,
        predicted_change_pct: float
    ) -> int:
        """Persist next-trading-session forecast results."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO prediction_history (
                    symbol, model_name, last_known_date, last_close, predicted_next_close, predicted_change_pct, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (symbol, model_name, last_known_date, last_close, predicted_next_close, predicted_change_pct, datetime.now().isoformat())
            )
            conn.commit()
            return cursor.lastrowid

    def get_predictions(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve forecast history entries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT * FROM prediction_history WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
                    (symbol, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM prediction_history ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                )
            return [dict(row) for row in cursor.fetchall()]

    def save_quality_report(
        self,
        symbol: str,
        total_rows: int,
        valid_rows: int,
        missing_values: int,
        duplicate_rows: int,
        report_dict: Dict[str, Any]
    ) -> int:
        """Save a snapshot of dataset quality validation checks."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO data_quality_reports (
                    symbol, total_rows, valid_rows, missing_values, duplicate_rows, report_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (symbol, total_rows, valid_rows, missing_values, duplicate_rows, json.dumps(report_dict), datetime.now().isoformat())
            )
            conn.commit()
            return cursor.lastrowid

    def get_recent_ingestions(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Fetch recent ingestion logs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM data_ingestion_log ORDER BY fetch_time DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]


# Singleton default instance
db = DatabaseManager()
