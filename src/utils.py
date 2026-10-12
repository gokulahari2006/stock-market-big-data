"""Utility functions and logging helpers for the Stock Market Big Data platform."""

import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime, date
from typing import Generator, Any, Optional

# Setup standard logger
logger = logging.getLogger("StockAnalytics")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@contextmanager
def execution_timer(task_name: str = "Task") -> Generator[dict, None, None]:
    """Context manager to accurately measure execution time using high-resolution perf_counter.

    Yields a dictionary with the elapsed time upon block exit.
    """
    stats = {"task": task_name, "start": 0.0, "end": 0.0, "elapsed_sec": 0.0}
    start_time = time.perf_counter()
    stats["start"] = start_time
    logger.info(f"Starting {task_name}...")
    try:
        yield stats
    finally:
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        stats["end"] = end_time
        stats["elapsed_sec"] = elapsed
        logger.info(f"Completed {task_name} in {elapsed:.4f} seconds")


def format_currency(value: Optional[float], currency_symbol: str = "₹") -> str:
    """Format floating point numbers into Indian Rupee or general currency string."""
    if value is None or (isinstance(value, float) and (value != value)):  # NaN check
        return "N/A"
    return f"{currency_symbol}{value:,.2f}"


def format_percentage(value: Optional[float]) -> str:
    """Format decimal percentage into human readable string with +/- sign."""
    if value is None or (isinstance(value, float) and (value != value)):
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def format_volume(volume: Optional[int]) -> str:
    """Format large trade volume into readable short strings (e.g. 1.25M, 450K, 3.2B)."""
    if volume is None or (isinstance(volume, float) and (volume != volume)):
        return "N/A"
    abs_vol = abs(volume)
    if abs_vol >= 1_000_000_000:
        return f"{volume / 1_000_000_000:.2f}B"
    if abs_vol >= 1_000_000:
        return f"{volume / 1_000_000:.2f}M"
    if abs_vol >= 1_000:
        return f"{volume / 1_000:.2f}K"
    return str(volume)


def safe_date_str(val: Any) -> str:
    """Convert various date representations safely to YYYY-MM-DD string."""
    if isinstance(val, (datetime, date)):
        return val.strftime("%Y-%m-%d")
    return str(val)[:10]
