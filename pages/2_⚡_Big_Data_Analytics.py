"""Big Data Analytics page demonstrating Apache Spark (PySpark) DataFrame transformations,
distributed window functions, Parquet exports, and empirical Pandas vs PySpark benchmarking.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from src.spark_processor import spark_engine
from src.data_collection import collector
from src.config import SUPPORTED_TICKERS, PROCESSED_DATA_DIR
from src.visualization import create_benchmark_chart

st.set_page_config(page_title="Big Data Analytics", page_icon="⚡", layout="wide")

st.title("⚡ Big Data Analytics with Apache Spark")
st.markdown("##### *Distributed Data Processing, Window Functions, and Parquet Storage Pipeline*")

# Engine Status Card
col_status1, col_status2 = st.columns([0.7, 0.3])
with col_status1:
    if spark_engine.spark_available and spark_engine.spark is not None:
        st.success("🟢 **Apache Spark Engine Active:** PySpark Local Cluster Ready (Multi-threaded JVM).")
    else:
        st.warning(
            f"🟡 **Spark Fallback Active:** Running via Pandas engine. "
            f"Reason: {spark_engine.initialization_error or 'Java/PySpark environment not initialized.'}"
        )

with col_status2:
    st.info(f"**Storage Format:** Apache Parquet (Columnar)\n\nTarget: `{PROCESSED_DATA_DIR.name}/`")

st.divider()

# Section 1: Multi-Stock Spark Pipeline Execution
st.subheader("1. Execute Big Data Pipeline Across Equities")
st.markdown(
    "This pipeline ingests historical data for multiple equities, constructs PySpark DataFrames, "
    "validates schemas using `StructType`, performs window-based lagging and rolling aggregations, "
    "and persists the transformed data into Parquet columnar format."
)

tickers_to_process = st.multiselect(
    "Select Equities to Ingest into Spark Pipeline",
    options=list(SUPPORTED_TICKERS.keys()),
    default=["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
)

if st.button("🚀 Run PySpark Analytics Pipeline", type="primary", use_container_width=True):
    with st.spinner("Ingesting and processing through PySpark pipeline..."):
        all_dfs = []
        for sym in tickers_to_process:
            df_sym, _, _ = collector.fetch_stock_data(symbol=sym, period="2y")
            df_sym["symbol"] = sym
            all_dfs.append(df_sym)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        results = spark_engine.run_spark_pipeline(combined_df)
        st.session_state["spark_pipeline_results"] = results

if "spark_pipeline_results" in st.session_state:
    res = st.session_state["spark_pipeline_results"]

    st.success(f"Pipeline executed via **{res['engine']}** in **{res['duration_seconds']} seconds**.")

    # Execution KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Input Records Ingested", f"{res['input_records']:,}")
    with kpi2:
        st.metric("Clean Output Records", f"{res['output_records']:,}")
    with kpi3:
        st.metric("Symbols Aggregated", f"{res['symbols_processed']}")
    with kpi4:
        st.metric("Processing Time", f"{res['duration_seconds']:.4f} s")

    # Stock-wise Distributed Aggregation Results
    st.markdown("#### 🏢 Distributed Stock-Wise Aggregations (`groupBy('symbol')`)")
    st.dataframe(res["stock_aggregations"], use_container_width=True)

    # Date-wise Market Summary
    st.markdown("#### 📅 Date-Wise Market Aggregate Price & Volume (`groupBy('date_parsed')`)")
    st.dataframe(res["date_aggregations"].tail(20), use_container_width=True)

    # Parquet Export Notice
    if res.get("parquet_saved"):
        st.info("✅ Processed dataset successfully exported to `data/processed/spark_analytics_output.parquet`.")

st.divider()

# Section 2: Empirical Pandas vs PySpark Benchmark
st.subheader("2. Pandas vs Apache Spark Performance Benchmark")
st.markdown(
    "Evaluate execution times between in-memory single-threaded Pandas and Apache Spark's DataFrame engine. "
    "Both engines perform identical transformations: schema casting, filtering, window lags, moving averages, "
    "rolling volatility, and multi-level aggregations on a reproducible test dataset."
)

col_bench_ctrl, col_bench_info = st.columns([0.4, 0.6])
with col_bench_ctrl:
    bench_records = st.select_slider(
        "Select Benchmark Dataset Record Count",
        options=[10000, 25000, 50000, 100000],
        value=50000
    )
    run_bench = st.button("⏱️ Run Benchmark Comparison", use_container_width=True)

with col_bench_info:
    st.caption(
        "**Technical Principle for College Viva:**\n\n"
        "- **Pandas:** Optimized C/NumPy arrays in single-node RAM. Near-instantaneous on small/medium datasets (< 500k rows) with zero JVM initialization overhead.\n"
        "- **Apache Spark:** Distributed memory engine. Incurs JVM startup, Catalyst optimizer planning, and IPC overhead on small data, but scales linearly to multi-node clusters handling billions of records."
    )

if run_bench:
    with st.spinner(f"Running benchmark on {bench_records:,} synthetic records across both engines..."):
        bench_res = spark_engine.benchmark_pandas_vs_spark(num_records=bench_records)
        st.session_state["bench_results"] = bench_res

if "bench_results" in st.session_state:
    b_res = st.session_state["bench_results"]

    p_time = b_res["pandas_time_seconds"]
    s_time = b_res["spark_time_seconds"]

    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.metric("Pandas Runtime", f"{p_time:.4f} s")
    with col_res2:
        st.metric("PySpark Runtime", f"{s_time:.4f} s" if s_time else "N/A (Local fallback)")
    with col_res3:
        st.metric("Results Numerically Agree", "✅ True (< 0.05 tolerance)" if b_res["results_agree"] else "❌ Discrepancy")

    # Benchmark Visualization
    fig_bench = create_benchmark_chart(p_time, s_time, b_res["total_benchmark_records"])
    st.plotly_chart(fig_bench, use_container_width=True)

    st.info(f"💡 **Benchmark Insight:** {b_res['observation']}")
