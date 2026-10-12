"""Multi-Stock Comparison page: Base-100 normalized performance charts,
return correlation matrices, volatility profiles, and comparative metrics across multiple equities.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np

from src.config import SUPPORTED_TICKERS, DATE_PRESETS
from src.data_collection import collector
from src.preprocessing import DataPreprocessor
from src.financial_analytics import FinancialAnalytics
from src.visualization import (
    create_normalized_performance_chart,
    create_correlation_heatmap
)

st.set_page_config(page_title="Multi-Stock Comparison", page_icon="⚖️", layout="wide")

st.title("⚖️ Multi-Stock Performance Comparison")
st.markdown("##### *Normalized Comparative Analytics, Cross-Asset Correlations & Risk Profiles*")

# Selection controls
selected_tickers = st.multiselect(
    "Select Equities to Compare",
    options=list(SUPPORTED_TICKERS.keys()),
    default=["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
)

selected_preset = st.selectbox(
    "Comparison Time Horizon",
    options=list(DATE_PRESETS.keys()),
    index=3  # 1y
)
period_code = DATE_PRESETS[selected_preset]

if len(selected_tickers) < 2:
    st.info("Please select at least 2 stocks to enable comparative analytics.")
    st.stop()

# Fetch and combine datasets
dfs = []
summary_rows = []
returns_dict = {}

with st.spinner("Retrieving and standardizing historical datasets..."):
    for sym in selected_tickers:
        raw_df, _, _ = collector.fetch_stock_data(symbol=sym, period=period_code)
        clean_df = DataPreprocessor.clean_stock_data(raw_df, symbol=sym)
        df_ind = FinancialAnalytics.calculate_all_indicators(clean_df)
        df_ind["symbol"] = sym
        dfs.append(df_ind)

        summ = FinancialAnalytics.get_market_summary(df_ind)
        summary_rows.append({
            "Symbol": sym,
            "Company": SUPPORTED_TICKERS.get(sym, {}).get("name", sym),
            "Latest Close (₹)": summ.get("latest_close", 0.0),
            "Period Return (%)": f"{summ.get('period_return_pct', 0.0):+.2f}%",
            "Period High (₹)": summ.get("period_high", 0.0),
            "Period Low (₹)": summ.get("period_low", 0.0),
            "Annualized Volatility (%)": f"{summ.get('annualized_volatility_pct', 0.0):.2f}%",
            "Avg Daily Volume": f"{int(summ.get('avg_volume', 0)):,}"
        })

        # Align daily returns for correlation
        returns_dict[sym] = df_ind.set_index("date")["daily_return_pct"]

multi_df = pd.concat(dfs, ignore_index=True)

# 1. Base-100 Normalized Performance Chart
st.subheader("📈 Normalized Cumulative Performance (Base = 100)")
st.caption(
    "Comparing raw stock prices directly is misleading because absolute price levels depend on share split history "
    "and market capitalization. Normalizing all series to a common starting value of 100 reveals true relative growth."
)
fig_norm = create_normalized_performance_chart(multi_df)
st.plotly_chart(fig_norm, use_container_width=True)

# 2. Comparative Statistics Table
st.subheader("📊 Comparative Risk & Return Summary")
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

# 3. Cross-Asset Return Correlation Heatmap
st.divider()
st.subheader("🔥 Cross-Asset Daily Return Correlation Matrix (Pearson)")
st.caption(
    "Measures how closely the daily returns of two stocks move together on a scale from -1.0 to +1.0. "
    "A lower correlation indicates better portfolio diversification benefits."
)

returns_df = pd.DataFrame(returns_dict).dropna()
if len(returns_df) > 5:
    corr_matrix = returns_df.corr()
    fig_corr = create_correlation_heatmap(corr_matrix)
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.warning("Insufficient overlapping trading dates to compute correlation matrix.")
