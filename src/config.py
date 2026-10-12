"""Configuration settings for Stock Market Big Data Analytics & Machine Learning Project.
Centralizes directory paths, ticker catalogs, technical indicator parameters,
machine learning settings, and UI visual styling.
"""

from pathlib import Path
from typing import Dict, List

# -----------------------------------------------------------------------------
# Base Directories & Storage Paths
# -----------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"
MODELS_DIR: Path = BASE_DIR / "models"
DATABASE_DIR: Path = BASE_DIR / "database"
DB_PATH: Path = DATABASE_DIR / "stock_analytics.db"

# Ensure all vital directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, MODELS_DIR, DATABASE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Supported Stock Tickers (NSE India Yahoo Finance Tickers)
# -----------------------------------------------------------------------------
SUPPORTED_TICKERS: Dict[str, Dict[str, str]] = {
    "RELIANCE.NS": {
        "name": "Reliance Industries Ltd",
        "sector": "Energy & Telecom",
        "description": "India's largest conglomerate by market cap, active in refining, petrochemicals, retail, and digital services."
    },
    "TCS.NS": {
        "name": "Tata Consultancy Services",
        "sector": "Information Technology",
        "description": "Global IT services and consulting powerhouse, part of the Tata Group."
    },
    "INFY.NS": {
        "name": "Infosys Ltd",
        "sector": "Information Technology",
        "description": "Leading provider of next-generation digital services and consulting."
    },
    "HDFCBANK.NS": {
        "name": "HDFC Bank Ltd",
        "sector": "Financial Services (Banking)",
        "description": "India's largest private sector bank by assets and market capitalization."
    },
    "ICICIBANK.NS": {
        "name": "ICICI Bank Ltd",
        "sector": "Financial Services (Banking)",
        "description": "Prominent private sector bank offering retail, corporate, and investment banking."
    },
    "SBIN.NS": {
        "name": "State Bank of India",
        "sector": "Financial Services (Public Banking)",
        "description": "India's largest public sector commercial bank and financial services statutory body."
    },
    "ITC.NS": {
        "name": "ITC Ltd",
        "sector": "Fast Moving Consumer Goods (FMCG)",
        "description": "Multi-business conglomerate with presence in FMCG, hotels, paperboards, and agri-business."
    },
    "LT.NS": {
        "name": "Larsen & Toubro Ltd",
        "sector": "Engineering & Construction",
        "description": "Major multinational conglomerate in engineering, procurement, construction, and manufacturing."
    },
    "TATAMOTORS.NS": {
        "name": "Tata Motors Ltd",
        "sector": "Automobile",
        "description": "Leading global automobile manufacturer including passenger cars, commercial vehicles, and Jaguar Land Rover."
    },
    "BHARTIARTL.NS": {
        "name": "Bharti Airtel Ltd",
        "sector": "Telecommunications",
        "description": "One of India's leading telecommunications service providers operating in 18 countries."
    }
}

DEFAULT_TICKER: str = "RELIANCE.NS"
DEFAULT_PERIOD: str = "1y"

DATE_PRESETS: Dict[str, str] = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "Custom Date Range": "custom"
}

# -----------------------------------------------------------------------------
# Financial Analytics & Indicator Configuration
# -----------------------------------------------------------------------------
SMA_WINDOWS: List[int] = [5, 10, 20, 50, 100, 200]
EMA_WINDOWS: List[int] = [12, 26]
VOLATILITY_WINDOW: int = 20
RSI_WINDOW: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
BOLLINGER_WINDOW: int = 20
BOLLINGER_STD_DEV: float = 2.0

# -----------------------------------------------------------------------------
# Machine Learning Feature & Training Parameters
# -----------------------------------------------------------------------------
TRAIN_TEST_SPLIT_RATIO: float = 0.80  # Chronological 80% train, 20% test
RANDOM_STATE: int = 42

FEATURE_COLUMNS: List[str] = [
    "close_lag_1",
    "close_lag_2",
    "close_lag_3",
    "close_lag_5",
    "close_lag_10",
    "close_lag_20",
    "return_lag_1",
    "return_lag_5",
    "return_lag_10",
    "volume_lag_1",
    "volume_lag_5",
    "sma_5",
    "sma_10",
    "sma_20",
    "ema_12",
    "ema_26",
    "volatility_20",
    "rsi_14",
    "macd",
    "macd_signal"
]

TARGET_COLUMN: str = "target_next_close"

MODEL_HYPERPARAMETERS = {
    "RandomForest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },
    "GradientBoosting": {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 4,
        "random_state": RANDOM_STATE
    }
}

# -----------------------------------------------------------------------------
# Big Data / PySpark Configuration
# -----------------------------------------------------------------------------
SPARK_APP_NAME: str = "StockMarketBigDataAnalytics"
SPARK_MASTER: str = "local[*]"
SPARK_DRIVER_MEMORY: str = "2g"

# -----------------------------------------------------------------------------
# Dark Financial Theme UI Colors & Styling
# -----------------------------------------------------------------------------
THEME_COLORS = {
    "background": "#0e1117",
    "card_background": "#161b22",
    "card_border": "#30363d",
    "text_primary": "#f0f6fc",
    "text_secondary": "#8b949e",
    "accent_green": "#00c076",
    "accent_red": "#ff4b4b",
    "accent_blue": "#2962ff",
    "accent_purple": "#8a2be2",
    "accent_amber": "#ffab00",
    "grid_color": "#21262d"
}

# -----------------------------------------------------------------------------
# Academic & Legal Disclaimers
# -----------------------------------------------------------------------------
PROJECT_DISCLAIMER: str = (
    "DISCLAIMER: This application is developed strictly for educational, academic, "
    "and research purposes as part of a college major project. Stock market predictions "
    "are statistical approximations based on historical data and do NOT guarantee future "
    "performance or market profits. They must not be construed as investment or trading advice. "
    "yfinance is an unofficial API subject to Yahoo Finance terms and availability."
)
