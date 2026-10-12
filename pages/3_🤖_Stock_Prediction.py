"""Stock Price Prediction page: Machine learning forecasting for next-day close,
chronological test evaluation, model comparison tables, and residual analysis.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np

from src.config import SUPPORTED_TICKERS, DEFAULT_TICKER, PROJECT_DISCLAIMER
from src.data_collection import collector
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.prediction_model import StockPredictor
from src.visualization import create_prediction_comparison_chart, create_residuals_chart
from src.utils import format_currency

st.set_page_config(page_title="Stock Prediction", page_icon="🤖", layout="wide")

st.title("🤖 Next-Session Stock Price Prediction")
st.markdown("##### *Predicting the Next Trading Day's Closing Price Using Supervised Machine Learning*")

# Stock Selector
selected_ticker = st.selectbox(
    "Select Stock Ticker for Predictive Modeling",
    options=list(SUPPORTED_TICKERS.keys()),
    index=0,
    format_func=lambda t: f"{t} ({SUPPORTED_TICKERS[t]['name']})"
)

# Fetch, preprocess, and engineer features
with st.spinner("Preparing engineered feature matrix..."):
    raw_df, source_label, is_sample = collector.fetch_stock_data(symbol=selected_ticker, period="2y")
    clean_df = DataPreprocessor.clean_stock_data(raw_df, symbol=selected_ticker)
    df_features = FeatureEngineer.create_features(clean_df)
    X, y, latest_features = FeatureEngineer.get_modeling_data(df_features)

st.caption(
    f"Dataset: **{len(X):,}** historical sessions with verified feature-target pairs | "
    f"Data Source: **{source_label}**"
)

# Initialize Predictor in session state
if "predictor" not in st.session_state:
    st.session_state["predictor"] = StockPredictor()

predictor: StockPredictor = st.session_state["predictor"]

# Training & Evaluation Section
col_train_btn, col_train_info = st.columns([0.3, 0.7])
with col_train_btn:
    train_clicked = st.button("🚀 Train & Evaluate Models (80/20 Split)", type="primary", use_container_width=True)

with col_train_info:
    st.markdown(
        "**Strict Time-Series Protocol:** Data is split chronologically into the earliest 80% (Training) "
        "and latest 20% (Testing). Features are lagged strictly up to session *t* to eliminate lookahead bias."
    )

if train_clicked or f"eval_{selected_ticker}" not in st.session_state:
    with st.spinner("Training Persistence Baseline, Linear Regression, Random Forest, and Gradient Boosting..."):
        eval_data = predictor.train_and_evaluate_all(X, y, symbol=selected_ticker)
        st.session_state[f"eval_{selected_ticker}"] = eval_data

eval_data = st.session_state[f"eval_{selected_ticker}"]
results = eval_data["results"]
best_model = eval_data["best_model"]

st.divider()

# Section 1: Next-Session Prediction Card
st.subheader("🔮 Next Trading Session Forecast")
col_model_select, col_pred_btn = st.columns([0.6, 0.4])

with col_model_select:
    chosen_model = st.selectbox(
        "Select Model for Next-Session Forecast",
        options=["Random Forest", "Linear Regression", "Gradient Boosting", "Persistence Baseline"],
        index=0
    )

with col_pred_btn:
    st.write("")
    st.write("")
    predict_clicked = st.button("⚡ Generate Next-Session Estimate", use_container_width=True)

forecast_result = predictor.predict_next_session(
    latest_features=latest_features,
    model_name=chosen_model,
    symbol=selected_ticker
)

# Display Forecast KPI Cards
card1, card2, card3, card4 = st.columns(4)
latest_known_close = forecast_result["latest_close"]
pred_close = forecast_result["predicted_next_close"]
abs_diff = forecast_result["absolute_change"]
pct_diff = forecast_result["percentage_change"]

diff_color = "#00c076" if abs_diff >= 0 else "#ff4b4b"
sign = "+" if abs_diff >= 0 else ""

with card1:
    st.metric("Latest Known Close", format_currency(latest_known_close))
with card2:
    st.metric("Estimated Next-Session Close", format_currency(pred_close), f"{sign}{abs_diff:,.2f} ({sign}{pct_diff:.2f}%)")
with card3:
    st.metric("Active Model", chosen_model)
with card4:
    st.metric("Recommended Model", best_model, "Lowest Test RMSE")

st.divider()

# Section 2: Model Performance Comparison Table
st.subheader("📊 Model Performance Comparison (Held-Out Test Set)")
st.caption("Evaluated on unseen historical test data (20% chronological split).")

comparison_rows = []
for m_name, m_info in results.items():
    metrics = m_info["metrics"]
    comparison_rows.append({
        "Model": m_name,
        "MAE (₹)": metrics["mae"],
        "RMSE (₹)": metrics["rmse"],
        "MAPE (%)": f"{metrics['mape']:.2f}%",
        "Directional Accuracy (%)": f"{metrics['directional_accuracy']:.2f}%",
        "Status": "⭐ Recommended" if m_name == best_model else "Evaluated"
    })

comparison_df = pd.DataFrame(comparison_rows)
st.dataframe(comparison_df, use_container_width=True)

# Section 3: Visual Evaluation (Actual vs Predicted)
st.subheader("📈 Out-of-Sample Actual vs Predicted Prices")
preds_for_chart = {m: results[m]["predictions"] for m in results}
fig_eval = create_prediction_comparison_chart(
    y_test=eval_data["y_test"],
    predictions_dict=preds_for_chart
)
st.plotly_chart(fig_eval, use_container_width=True)

# Section 4: Residual Error Diagnostics
st.subheader(f"🔍 Residual Error Diagnostics ({chosen_model})")
if chosen_model in results:
    model_preds = results[chosen_model]["predictions"]
    fig_residuals = create_residuals_chart(eval_data["y_test"], model_preds, chosen_model)
    st.plotly_chart(fig_residuals, use_container_width=True)

st.divider()
st.warning(
    f"⚠️ **Limitations & Disclaimer:** {PROJECT_DISCLAIMER} "
    f"External macroeconomic catalysts, geopolitical shifts, earnings releases, and sudden liquidity shocks "
    f"cannot be fully anticipated by autoregressive price patterns alone."
)
