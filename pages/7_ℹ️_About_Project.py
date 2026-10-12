"""About Project page: Academic viva preparation guide, system architecture,
methodological foundations, big data concepts, and future scopes.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from src.config import PROJECT_DISCLAIMER

st.set_page_config(page_title="About Project", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About the Project & Academic Documentation")
st.markdown("##### *Stock Market Data Analysis and Prediction Using Big Data Analytics and Machine Learning*")

st.divider()

# Abstract & Problem Statement
col_abs, col_prob = st.columns(2)
with col_abs:
    st.markdown("""
    ### 🎯 Project Abstract
    This college major project develops a comprehensive, scalable financial analytics platform
    capable of ingesting multi-year historical stock market equity data, performing big data
    transformations using **Apache Spark (PySpark)**, engineering predictive lag-based features
    without lookahead data leakage, and predicting the next trading session's closing price
    using supervised machine learning (**Random Forest**, **Linear Regression**, and **Gradient Boosting**).
    The system is encapsulated within a responsive **Streamlit** dashboard featuring institutional-grade
    dark **Plotly** visualizations, SQLite audit persistence, and columnar **Parquet** storage.
    """)

with col_prob:
    st.markdown("""
    ### ⚠️ Problem Statement
    Retail financial data analysis frequently suffers from:
    1. **Data Leakage:** Naive predictive models often inadvertently train on future session indicators, producing deceptively high test accuracy that fails in real trading.
    2. **Scalability Bottlenecks:** Traditional Pandas in-memory analysis struggles when scaled across thousands of equities and intraday tick streams.
    3. **Misleading Baseline Comparisons:** Machine learning models are often claimed to be accurate without benchmarking against a naive persistence baseline (today's close).
    4. **Fragmented Tooling:** Disconnect between data ingestion, big data distributed processing, machine learning inference, and interactive visualization.
    """)

st.divider()

# System Architecture & Flow
st.subheader("🏛️ System Architecture & Data Flow")
st.markdown("""
```
+-----------------------------------------------------------------------------------+
|                            DATA COLLECTION LAYER                                  |
|   yfinance Live API (NSE Tickers)  <--->  Local Cache CSV  <--->  Sample Fixtures |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        PREPROCESSING & VALIDATION LAYER                           |
|      Column Normalization  |  Timezone Standardization  |  Invariant Auditing     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                     BIG DATA ANALYTICS LAYER (APACHE SPARK)                       |
|   PySpark DataFrames | Window Functions (Lag, SMA, Volatility) | Parquet Storage  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                     FEATURE ENGINEERING & MACHINE LEARNING                        |
|  20 Causal Lag Features  |  Chronological 80/20 Split  |  Zero Lookahead Bias     |
|   Linear Regression  |  Random Forest Regressor  |  Gradient Boosting Booster     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       PRESENTATION & STORAGE LAYER                                |
|    Streamlit Multi-Page Web App  |  Plotly Visuals  |  SQLite Metadata Audit      |
+-----------------------------------------------------------------------------------+
```
""")

st.divider()

# Technology Stack Summary
st.subheader("💻 Core Technology Stack")
t1, t2, t3, t4 = st.columns(4)
with t1:
    st.markdown("""
    **Core & Web UI**
    - Python 3.11 / 3.12
    - Streamlit
    - Plotly Interactive Charts
    """)
with t2:
    st.markdown("""
    **Data Processing**
    - Apache Spark (PySpark 3.5)
    - Pandas & NumPy
    - PyArrow Columnar
    """)
with t3:
    st.markdown("""
    **Machine Learning**
    - Scikit-Learn
    - Random Forest Regressor
    - Linear Regression
    - Joblib Serialization
    """)
with t4:
    st.markdown("""
    **Storage & Testing**
    - Apache Parquet
    - SQLite Relational DB
    - Pytest Test Suite
    """)

st.divider()

# Viva Examination Q&A Highlights
st.subheader("🎓 Key Concepts for Project Viva Examination")

with st.expander("Q1: Why is this categorized as a Big Data project?"):
    st.markdown("""
    Financial markets generate millions of trades per second across global exchanges. While historical daily data for a single stock is compact, real-world market surveillance requires joining hundreds of equities across decades, tick-by-tick volumes, and order book states.
    By designing the pipeline with **Apache Spark (PySpark)** DataFrames, Window specifications (`Window.partitionBy().orderBy()`), and **Apache Parquet** columnar storage, the architecture is horizontally scalable and ready for multi-node Hadoop/Kubernetes clusters.
    """)

with st.expander("Q2: How is Target Leakage strictly prevented?"):
    st.markdown("""
    Target leakage occurs when an algorithm uses data during training that would not be available at the actual moment of prediction.
    In our system:
    1. Every feature at row $t$ is calculated strictly using observations up to session $t$.
    2. The target variable $y_t$ is the close price of session $t+1$.
    3. The dataset is chronologically split (earliest 80% train, latest 20% test) without shuffling.
    4. Preprocessing transformations (e.g. `StandardScaler`) are fit *only* on the training split, preventing mean/variance leakage from the test split.
    """)

with st.expander("Q3: What does Directional Accuracy measure, and why is it crucial?"):
    st.markdown("""
    Directional accuracy measures whether the predicted price change correctly anticipates the market's upward or downward move:
    $$\\text{Directional Accuracy} = \\frac{1}{N} \\sum_{i=1}^N \\mathbf{1}_{\\text{sign}(\\hat{y}_{t+1} - y_t) == \\text{sign}(y_{t+1} - y_t)} \\times 100\\%$$
    In quantitative finance, predicting the correct direction (sign) is often more valuable than achieving a slightly lower RMSE, as trading strategies depend on entering long (buy) or short (sell) positions based on anticipated directional trends.
    """)

with st.expander("Q4: Why can Pandas run faster than Apache Spark on small datasets?"):
    st.markdown("""
    Apache Spark carries JVM initialization overhead, Catalyst query plan optimization time, and inter-process communication (IPC) between Python and Java virtual machines (Py4J).
    On small datasets (< 100,000 rows), Pandas processes data in-place via optimized C/NumPy binaries with zero JVM overhead. However, when datasets exceed single-node RAM (gigabytes to terabytes), Pandas crashes with `MemoryError`, whereas Spark partitions data across cluster nodes with resilient distributed execution.
    """)

st.divider()

# Project Limitations & Future Enhancements
col_lim, col_fut = st.columns(2)
with col_lim:
    st.markdown("""
    ### ⚠️ Realistic Project Limitations
    1. **Efficient Market Hypothesis (EMH):** Stock prices incorporate public information rapidly; historical OHLCV prices alone cannot capture sudden exogenous events.
    2. **Absence of Alternative Data:** The current model does not ingest live news headlines, central bank rate decisions, or earnings call transcripts.
    3. **Daily Granularity:** Daily closing prices aggregate significant intraday order book volatility into a single number.
    """)

with col_fut:
    st.markdown("""
    ### 🚀 Future Enhancements
    1. **NLP Sentiment Analysis:** Ingesting financial news and social sentiment using FinBERT.
    2. **Deep Learning Sequence Models:** Incorporating LSTM (Long Short-Term Memory) networks and Temporal Fusion Transformers.
    3. **Real-Time Streaming Pipeline:** Integrating Apache Kafka for real-time tick streaming into Spark Structured Streaming.
    4. **Automated Backtesting Engine:** Simulating historical strategy execution with trading fees and slippage models.
    """)

st.divider()
st.caption(PROJECT_DISCLAIMER)
