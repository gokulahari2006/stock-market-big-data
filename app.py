"""Main Entry Point: Stock Market Big Data Analytics & Machine Learning Platform.
Home & Market Overview Dashboard.
"""

import sys
from pathlib import Path

# Add project root to sys.path so modules resolve cleanly
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from src.config import (
    SUPPORTED_TICKERS,
    DEFAULT_TICKER,
    DATE_PRESETS,
    DEFAULT_PERIOD,
    THEME_COLORS,
    PROJECT_DISCLAIMER
)
from src.data_collection import collector
from src.preprocessing import DataPreprocessor
from src.financial_analytics import FinancialAnalytics
from src.visualization import create_price_line_chart
from src.utils import format_currency, format_percentage, format_volume

# Page configuration
st.set_page_config(
    page_title="Stock Market Big Data Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark institutional finance aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0f6fc;
        margin-top: 4px;
    }
    .metric-delta-positive {
        font-size: 0.95rem;
        font-weight: 600;
        color: #00c076;
    }
    .metric-delta-negative {
        font-size: 0.95rem;
        font-weight: 600;
        color: #ff4b4b;
    }
    .pipeline-step {
        background: #161b22;
        border-left: 3px solid #2962ff;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


def render_sidebar():
    """Sidebar controls for stock selection, date presets, and data refresh."""
    st.sidebar.markdown("## 📊 **Market Terminal**")
    st.sidebar.caption("Big Data & ML Analytics Platform")
    st.sidebar.divider()

    # Stock selector
    ticker_options = list(SUPPORTED_TICKERS.keys())
    format_func = lambda t: f"{t} — {SUPPORTED_TICKERS[t]['name']}"
    selected_ticker = st.sidebar.selectbox(
        "Select Stock Ticker",
        options=ticker_options,
        index=ticker_options.index(DEFAULT_TICKER),
        format_func=format_func
    )

    # Date range preset
    selected_preset_label = st.sidebar.selectbox(
        "Historical Period",
        options=list(DATE_PRESETS.keys()),
        index=3  # Default 1 Year
    )
    period_code = DATE_PRESETS[selected_preset_label]

    start_date, end_date = None, None
    if period_code == "custom":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        with col2:
            end_date = st.date_input("End Date", value=datetime.now()).strftime("%Y-%m-%d")

    force_refresh = st.sidebar.button("🔄 Refresh Market Data", use_container_width=True)

    st.sidebar.divider()
    st.sidebar.info(
        f"**Selected Company:**\n\n"
        f"**{SUPPORTED_TICKERS[selected_ticker]['name']}**\n\n"
        f"*{SUPPORTED_TICKERS[selected_ticker]['sector']}*\n\n"
        f"{SUPPORTED_TICKERS[selected_ticker]['description']}"
    )

    return selected_ticker, period_code, start_date, end_date, force_refresh


def main():
    selected_ticker, period_code, start_date, end_date, force_refresh = render_sidebar()

    # Data Retrieval
    with st.spinner(f"Retrieving market data for {selected_ticker}..."):
        raw_df, source_label, is_sample = collector.fetch_stock_data(
            symbol=selected_ticker,
            period=period_code,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )

    # Clean & Preprocess
    try:
        clean_df = DataPreprocessor.clean_stock_data(raw_df, symbol=selected_ticker)
        analytics_df = FinancialAnalytics.calculate_all_indicators(clean_df)
        summary = FinancialAnalytics.get_market_summary(analytics_df)
    except Exception as e:
        st.error(f"Error processing historical data: {str(e)}")
        st.stop()

    # Header section
    st.title("📈 Stock Market Data Analysis & Prediction")
    st.markdown("##### *Big Data Analytics & Machine Learning Academic Major Project*")

    # Source & Status Banner
    badge_color = "warning" if is_sample else "success"
    if is_sample:
        st.warning(f"⚠️ **Demonstration Mode:** {source_label}. Real market download was bypassed or unavailable.")
    else:
        st.success(f"🟢 **Data Source:** {source_label} | Total Trading Sessions: {len(analytics_df):,}")

    # Top KPI Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)

    latest_price = summary.get("latest_close", 0.0)
    abs_change = summary.get("absolute_change", 0.0)
    pct_change = summary.get("percentage_change", 0.0)

    delta_color = "metric-delta-positive" if abs_change >= 0 else "metric-delta-negative"
    sign = "+" if abs_change >= 0 else ""

    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Latest Close</div>
            <div class="metric-value">{format_currency(latest_price)}</div>
            <div class="{delta_color}">{sign}{abs_change:,.2f} ({sign}{pct_change:.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Period High</div>
            <div class="metric-value">{format_currency(summary.get("period_high"))}</div>
            <div style="font-size:0.85rem; color:#8b949e;">Peak in period</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Period Low</div>
            <div class="metric-value">{format_currency(summary.get("period_low"))}</div>
            <div style="font-size:0.85rem; color:#8b949e;">Trough in period</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Daily Volume</div>
            <div class="metric-value">{format_volume(int(summary.get("avg_volume", 0)))}</div>
            <div style="font-size:0.85rem; color:#8b949e;">Shares traded/day</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Annualized Volatility</div>
            <div class="metric-value">{summary.get("annualized_volatility_pct", 0.0):.2f}%</div>
            <div style="font-size:0.85rem; color:#8b949e;">20-Day Rolling Std</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Period Return</div>
            <div class="metric-value">{summary.get("period_return_pct", 0.0):+.2f}%</div>
            <div style="font-size:0.85rem; color:#8b949e;">Cumulative return</div>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Closing Price Chart
    st.subheader(f"📊 {selected_ticker} — Historical Closing Price & Trading Volume")
    fig_overview = create_price_line_chart(analytics_df, selected_ticker)
    st.plotly_chart(fig_overview, use_container_width=True)

    # System Architecture & Data Pipeline Summary
    st.divider()
    st.subheader("🛠️ End-to-End System Pipeline")

    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    with p_col1:
        st.markdown("""
        <div class="pipeline-step">
            <b>1. Data Collection</b><br>
            <small>yfinance API & cached offline fallback for Indian NSE stocks.</small>
        </div>
        """, unsafe_allow_html=True)
    with p_col2:
        st.markdown("""
        <div class="pipeline-step">
            <b>2. Validation & Clean</b><br>
            <small>OHLC logic checks, deduplication, chronological date sorting.</small>
        </div>
        """, unsafe_allow_html=True)
    with p_col3:
        st.markdown("""
        <div class="pipeline-step">
            <b>3. Big Data Engine</b><br>
            <small>Apache Spark (PySpark) DataFrame transformations, window lag & Parquet.</small>
        </div>
        """, unsafe_allow_html=True)
    with p_col4:
        st.markdown("""
        <div class="pipeline-step">
            <b>4. Machine Learning</b><br>
            <small>Random Forest & Linear Regression with zero future data leakage.</small>
        </div>
        """, unsafe_allow_html=True)
    with p_col5:
        st.markdown("""
        <div class="pipeline-step">
            <b>5. Analytics Dashboard</b><br>
            <small>Interactive Plotly visualization with institutional dark theme.</small>
        </div>
        """, unsafe_allow_html=True)

    # Quick Navigation Cards
    st.divider()
    st.subheader("🚀 Navigate Modules")
    nav1, nav2, nav3, nav4 = st.columns(4)
    with nav1:
        st.markdown("### 📈 Stock Explorer")
        st.caption("Deep technical indicators (SMA, EMA, Bollinger Bands, RSI, MACD) and Candlestick charts.")
    with nav2:
        st.markdown("### ⚡ Big Data Analytics")
        st.caption("Execute PySpark transformations, distributed aggregations, and Pandas vs Spark runtime benchmarks.")
    with nav3:
        st.markdown("### 🤖 Stock Prediction")
        st.caption("Train ML models chronologically and forecast the next trading session closing price.")
    with nav4:
        st.markdown("### ⚖️ Multi-Stock Compare")
        st.caption("Compare multiple equities with base-100 normalized performance and correlation heatmaps.")

    st.caption(f"**Latest Historical Record Date:** {summary.get('latest_date')} | Delayed Daily Observations")
    st.caption(PROJECT_DISCLAIMER)


if __name__ == "__main__":
    main()
