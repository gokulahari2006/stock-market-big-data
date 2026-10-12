"""Stock Market Explorer: Interactive Candlestick charts, moving average overlays,
technical indicators (RSI, MACD, Bollinger Bands), and tabular CSV export.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from src.config import SUPPORTED_TICKERS, DEFAULT_TICKER, DATE_PRESETS
from src.data_collection import collector
from src.preprocessing import DataPreprocessor
from src.financial_analytics import FinancialAnalytics
from src.visualization import (
    create_candlestick_chart,
    create_ma_overlay_chart,
    create_bollinger_chart,
    create_rsi_chart,
    create_macd_chart,
    create_volatility_chart
)

st.set_page_config(page_title="Stock Explorer", page_icon="📈", layout="wide")

st.title("📈 Stock Market Explorer")
st.markdown("##### *Interactive Technical Analysis & Candlestick Visualization*")

# Sidebar Controls
st.sidebar.markdown("### 🛠️ Explorer Settings")
ticker_options = list(SUPPORTED_TICKERS.keys())
selected_ticker = st.sidebar.selectbox(
    "Choose Equity Ticker",
    options=ticker_options,
    index=ticker_options.index(DEFAULT_TICKER),
    format_func=lambda t: f"{t} ({SUPPORTED_TICKERS[t]['name']})"
)

selected_preset_label = st.sidebar.selectbox(
    "Historical Period",
    options=list(DATE_PRESETS.keys()),
    index=3  # 1y
)
period_code = DATE_PRESETS[selected_preset_label]

# Indicator Toggles
st.sidebar.markdown("### 🎛️ Indicator Overlays")
show_candlestick = st.sidebar.checkbox("Candlestick & Volume Chart", value=True)
show_mas = st.sidebar.checkbox("Moving Averages (SMA/EMA)", value=True)
show_bollinger = st.sidebar.checkbox("Bollinger Bands (20, 2σ)", value=False)
show_rsi = st.sidebar.checkbox("Relative Strength Index (RSI 14)", value=True)
show_macd = st.sidebar.checkbox("MACD Oscillator (12, 26, 9)", value=True)
show_volatility = st.sidebar.checkbox("Rolling 20-Day Volatility", value=False)

# Fetch and Process Data
with st.spinner("Loading stock data..."):
    raw_df, source_label, is_sample = collector.fetch_stock_data(
        symbol=selected_ticker,
        period=period_code
    )
    clean_df = DataPreprocessor.clean_stock_data(raw_df, symbol=selected_ticker)
    df_analytics = FinancialAnalytics.calculate_all_indicators(clean_df)

if is_sample:
    st.warning(f"Displaying Sample Dataset for {selected_ticker} ({source_label}).")
else:
    st.caption(f"Data source: {source_label} | Total sessions: {len(df_analytics)}")

# 1. Candlestick Chart
if show_candlestick:
    st.subheader(f"🕯️ {selected_ticker} Candlestick Price Action")
    fig_candle = create_candlestick_chart(df_analytics, selected_ticker)
    st.plotly_chart(fig_candle, use_container_width=True)

# 2. Moving Average Overlays
if show_mas:
    st.subheader("📉 Moving Average Trend Analysis")
    st.caption(
        "Simple Moving Averages (SMA 20, 50, 200) smooth out daily price fluctuations to reveal prevailing trends. "
        "Exponential Moving Averages (EMA 12, 26) assign higher weight to recent sessions."
    )
    fig_ma = create_ma_overlay_chart(df_analytics, selected_ticker)
    st.plotly_chart(fig_ma, use_container_width=True)

# 3. Bollinger Bands
if show_bollinger:
    st.subheader("🎯 Bollinger Bands")
    st.caption(
        "Bollinger Bands consist of a 20-day SMA middle band and two outer bands placed 2 standard deviations away. "
        "They expand during volatile periods and contract during consolidation."
    )
    fig_bb = create_bollinger_chart(df_analytics, selected_ticker)
    st.plotly_chart(fig_bb, use_container_width=True)

# 4. Momentum & Oscillators: RSI & MACD
col1, col2 = st.columns(2)

with col1:
    if show_rsi:
        st.subheader("⚡ Momentum: RSI (14)")
        st.caption("RSI measures the speed and change of price movements on a 0-100 scale. Values >70 suggest overbought conditions; values <30 suggest oversold.")
        fig_rsi = create_rsi_chart(df_analytics, selected_ticker)
        st.plotly_chart(fig_rsi, use_container_width=True)

with col2:
    if show_macd:
        st.subheader("🌊 Trend & Momentum: MACD")
        st.caption("MACD subtracts the 26-day EMA from the 12-day EMA. The 9-day EMA acts as the signal line.")
        fig_macd = create_macd_chart(df_analytics, selected_ticker)
        st.plotly_chart(fig_macd, use_container_width=True)

if show_volatility:
    st.subheader("🌪️ Rolling Volatility")
    fig_vol = create_volatility_chart(df_analytics, selected_ticker)
    st.plotly_chart(fig_vol, use_container_width=True)

# Historical Data Table & CSV Export
st.divider()
st.subheader("📋 Historical Observations Data Table")

col_table, col_export = st.columns([0.8, 0.2])
with col_export:
    csv_data = df_analytics.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name=f"{selected_ticker}_historical_analytics.csv",
        mime="text/csv",
        use_container_width=True
    )

display_cols = ["date", "open", "high", "low", "close", "volume", "daily_return_pct", "sma_20", "rsi_14"]
available_display = [c for c in display_cols if c in df_analytics.columns]
st.dataframe(
    df_analytics[available_display].tail(50).sort_values(by="date", ascending=False),
    use_container_width=True,
    height=320
)
