"""Data Management page: Local dataset inventory, data quality audit reports,
Parquet & CSV exports, and SQLite ingestion audit trails.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from src.storage import storage
from src.database import db
from src.config import SUPPORTED_TICKERS, DEFAULT_TICKER, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data_collection import collector
from src.data_validation import DataValidator
from src.preprocessing import DataPreprocessor

st.set_page_config(page_title="Data Management", page_icon="💾", layout="wide")

st.title("💾 Data Management & Ingestion Audit")
st.markdown("##### *Local Storage Catalog, Data Quality Audits, and Parquet/CSV File Operations*")

tab_inventory, tab_audit, tab_logs = st.tabs([
    "📁 Dataset Inventory",
    "🔍 Data Quality Audit",
    "📜 Ingestion History Logs"
])

# Tab 1: Dataset Inventory
with tab_inventory:
    st.subheader("Local Storage Inventory")
    st.caption("Displays raw cached files, cleaned processed datasets, Parquet files, and sample fixtures.")

    datasets = storage.list_datasets()
    if datasets:
        st.dataframe(pd.DataFrame(datasets), use_container_width=True)
    else:
        st.info("No local datasets found yet.")

    st.divider()
    st.subheader("📥 Retrieve / Refresh Market Dataset")

    col_t, col_p, col_b = st.columns([0.4, 0.3, 0.3])
    with col_t:
        target_symbol = st.selectbox("Ticker Symbol", options=list(SUPPORTED_TICKERS.keys()), index=0)
    with col_p:
        target_period = st.selectbox("Period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    with col_b:
        st.write("")
        st.write("")
        refresh_btn = st.button("🔄 Fetch & Store Data", use_container_width=True)

    if refresh_btn:
        with st.spinner(f"Fetching data for {target_symbol}..."):
            df_retrieved, source_msg, is_samp = collector.fetch_stock_data(
                symbol=target_symbol,
                period=target_period,
                force_refresh=True
            )
            clean_df = DataPreprocessor.clean_stock_data(df_retrieved, symbol=target_symbol)
            # Export to processed
            base_out = PROCESSED_DATA_DIR / f"{target_symbol.replace('.', '_')}_processed"
            csv_p, parq_p = DataPreprocessor.export_processed(clean_df, base_out)
            st.success(f"Successfully stored {len(clean_df)} records for {target_symbol}. Source: {source_msg}")
            st.rerun()

# Tab 2: Data Quality Audit
with tab_audit:
    st.subheader("🔍 Deep Data Quality Diagnostic Audit")
    st.caption("Executes structural schema validation, duplicate detection, and price invariant logic checks.")

    audit_symbol = st.selectbox("Select Dataset to Audit", options=list(SUPPORTED_TICKERS.keys()), key="audit_select")
    if st.button("🚀 Run Quality Audit", type="primary"):
        raw_df, src, _ = collector.fetch_stock_data(symbol=audit_symbol, period="2y")
        report = DataValidator.audit_quality(raw_df, symbol=audit_symbol)

        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.metric("Total Records Ingested", f"{report['total_records']:,}")
        with q2:
            st.metric("Valid Clean Records", f"{report['valid_records']:,}")
        with q3:
            st.metric("Duplicate Dates", f"{report['duplicate_dates']}")
        with q4:
            st.metric("Price Invariant Violations", f"{report['price_invariants_broken']}")

        st.markdown("#### Audit Diagnostic Status")
        if report["is_valid"]:
            st.success("✅ Dataset PASSED all structural validation and price invariant checks.")
        else:
            st.error("❌ Dataset contains issues requiring cleaning or preprocessing.")

        if report["issues"]:
            st.warning("Identified Quality Observations:")
            for issue in report["issues"]:
                st.write(f"- {issue}")

        # Save report snapshot to SQLite
        db.save_quality_report(
            symbol=audit_symbol,
            total_rows=report["total_records"],
            valid_rows=report["valid_records"],
            missing_values=len(report["missing_values"]),
            duplicate_rows=report["duplicate_dates"],
            report_dict=report
        )

# Tab 3: Ingestion Logs
with tab_logs:
    st.subheader("📜 Ingestion Audit Trail (SQLite `data_ingestion_log`)")
    st.caption("Tracks timestamps, source origins, record quantities, and execution statuses.")

    logs = db.get_recent_ingestions(limit=30)
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True)
    else:
        st.info("No ingestion logs recorded yet.")
