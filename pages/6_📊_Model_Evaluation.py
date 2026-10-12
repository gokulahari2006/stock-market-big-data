"""Model Evaluation page: In-depth academic evaluation of machine learning models,
overfitting analysis, residual diagnostics, and SQLite historical metric records.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np

from src.config import SUPPORTED_TICKERS, DEFAULT_TICKER
from src.database import db
from src.data_collection import collector
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.prediction_model import StockPredictor
from src.visualization import create_prediction_comparison_chart, create_residuals_chart

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")

st.title("📊 Machine Learning Model Evaluation")
st.markdown("##### *Rigorous Out-of-Sample Performance Benchmarking & Residual Diagnostics*")

symbol = st.selectbox(
    "Select Stock Ticker to Inspect",
    options=list(SUPPORTED_TICKERS.keys()),
    index=0
)

# Fetch, preprocess, and engineer features
with st.spinner("Preparing dataset for evaluation..."):
    raw_df, _, _ = collector.fetch_stock_data(symbol=symbol, period="2y")
    clean_df = DataPreprocessor.clean_stock_data(raw_df, symbol=symbol)
    df_features = FeatureEngineer.create_features(clean_df)
    X, y, _ = FeatureEngineer.get_modeling_data(df_features)

predictor = StockPredictor()
eval_data = predictor.train_and_evaluate_all(X, y, symbol=symbol)
results = eval_data["results"]

# KPI Overview
e1, e2, e3, e4 = st.columns(4)
with e1:
    st.metric("Total Usable Sessions", f"{len(X):,}")
with e2:
    st.metric("Training Sessions (80%)", f"{eval_data['train_size']:,}")
with e3:
    st.metric("Test Sessions (20%)", f"{eval_data['test_size']:,}")
with e4:
    st.metric("Best Model Selected", eval_data["best_model"], "Lowest Test RMSE")

st.divider()

# Model Metric Comparison Table
st.subheader("1. Out-of-Sample Performance Metrics")
rows = []
for m_name, m_info in results.items():
    m = m_info["metrics"]
    rows.append({
        "Model Name": m_name,
        "Mean Absolute Error (MAE)": f"₹{m['mae']:.2f}",
        "Root Mean Squared Error (RMSE)": f"₹{m['rmse']:.2f}",
        "Mean Absolute % Error (MAPE)": f"{m['mape']:.2f}%",
        "Directional Accuracy": f"{m['directional_accuracy']:.2f}%"
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# Educational Explanation Section
st.divider()
st.subheader("2. Key Evaluation Concepts for Viva Examination")

col_edu1, col_edu2 = st.columns(2)
with col_edu1:
    st.markdown("""
    #### 🎯 Directional Accuracy Explained
    Directional accuracy evaluates whether the model correctly forecasted the **direction of price movement**
    (Up or Down) compared to the previous trading session's close:
    $$\\text{Directional Accuracy} = \\frac{1}{N} \\sum_{i=1}^N \\mathbf{1}_{\\text{sign}(\\hat{y}_{t+1} - y_t) == \\text{sign}(y_{t+1} - y_t)} \\times 100\\%$$
    - Even when a model has low MAE by closely tracking yesterday's price, its directional accuracy might hover near 50% (random guess).
    - Random Forest models leverage multi-feature interactions to capture momentum and reversal signals better than naive persistence.
    """)

with col_edu2:
    st.markdown("""
    #### ⚠️ Overfitting & Data Leakage Prevention
    - **Why Train Error < Test Error:** Models can easily memorize historical price paths. High training $R^2$ often deteriorates on unseen future test sets.
    - **Chronological Split vs Random K-Fold:** Standard K-Fold randomly shuffles observations, causing **lookahead data leakage** where future prices are used to predict past prices. We strictly enforce chronological 80% train / 20% test splitting.
    - **Persistence Baseline:** If a machine learning model cannot beat a naive persistence forecast (predicting tomorrow equals today), it possesses zero real predictive edge.
    """)

# Actual vs Predicted Plot
st.divider()
st.subheader("3. Actual vs Predicted Price Trajectory (Held-Out Test Period)")
preds_map = {m: results[m]["predictions"] for m in results}
fig_eval = create_prediction_comparison_chart(
    y_test=eval_data["y_test"],
    predictions_dict=preds_map
)
st.plotly_chart(fig_eval, use_container_width=True)

# SQLite Evaluation History Table
st.divider()
st.subheader("4. Historical Model Evaluation Records (SQLite `model_evaluations`)")
hist_evals = db.get_evaluations(symbol=symbol, limit=20)
if hist_evals:
    st.dataframe(pd.DataFrame(hist_evals), use_container_width=True)
else:
    st.info("No prior evaluation history found in SQLite database.")
