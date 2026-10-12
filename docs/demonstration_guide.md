# College Project Demonstration & Presentation Guide

## Step-by-Step Viva Demonstration Script

This guide outlines the recommended sequence to demonstrate the application to examiners and project reviewers.

---

### Step 1: Pre-Demonstration Verification
Before presenting, run the automated test suite to verify system integrity:
```powershell
pytest
```
*Talking Point:* "All 18+ automated test cases pass deterministically using synthetic fixtures, validating schema checks, price invariant logic, leakage prevention, and model evaluation metrics."

---

### Step 2: Launch the Web Platform
Launch the Streamlit web dashboard:
```powershell
streamlit run app.py
```
*Talking Point:* "The web application initializes with an institutional dark financial theme inspired by professional market terminals."

---

### Step 3: Page 1 — Home & Market Overview (`app.py`)
1. **Show KPI Metric Cards:** Point out Latest Close, 1-Day Change, Period High/Low, Avg Daily Volume, and Annualized Volatility.
2. **Show Interactive Price & Volume Chart:** Demonstrate Plotly hover tooltips and interactive zooming.
3. **Show Data Source Status:** Explain the live yfinance integration vs local cache vs verified sample fallback badge.
4. **Show Pipeline Overview:** Walk through the 5-stage architecture (Collection → Preprocessing → Big Data Analytics → Machine Learning → Dashboard).

---

### Step 4: Page 2 — Stock Market Explorer (`1_📈_Stock_Explorer.py`)
1. **Demonstrate Candlestick Chart:** Toggle OHLC Candlestick view and zoom into daily price action.
2. **Demonstrate Moving Average Overlays:** Show SMA 20, 50, 200 and EMA 12, 26.
3. **Demonstrate Oscillators:** Inspect RSI 14 (showing 30/70 oversold/overbought lines) and MACD (showing MACD line, signal line, and histogram).
4. **Download CSV:** Click "Download CSV" to demonstrate data export functionality.

---

### Step 5: Page 3 — Big Data Analytics (`2_⚡_Big_Data_Analytics.py`)
1. **Execute PySpark Pipeline:** Click "Run PySpark Analytics Pipeline" on multiple stocks (Reliance, TCS, Infosys, HDFC Bank, ICICI Bank).
2. **Show Distributed Transformations:** Point out window-based lagging, rolling standard deviation, and `groupBy('symbol')` aggregations.
3. **Show Parquet Columnar Output:** Explain how the output is persisted in Apache Parquet format.
4. **Run Pandas vs PySpark Benchmark:** Slide to 50,000 or 100,000 records and click "Run Benchmark Comparison".
   - *Crucial Viva Point:* Explain why Pandas is fast on small datasets due to zero JVM startup overhead, while Spark scales horizontally to distributed clusters on massive datasets.

---

### Step 6: Page 4 — Next-Session Stock Prediction (`3_🤖_Stock_Prediction.py`)
1. **Execute Model Training:** Click "Train & Evaluate Models (80/20 Split)".
2. **Explain Leakage Prevention:** Explain to the examiner that chronological splitting was strictly enforced without shuffling, and lag features prevent lookahead bias.
3. **Show Model Comparison Table:** Compare Persistence Baseline, Linear Regression, Random Forest, and Gradient Boosting across MAE, RMSE, MAPE, and Directional Accuracy.
4. **Generate Next-Day Estimate:** Select "Random Forest" and click "Generate Next-Session Estimate" to display the forecast card with absolute and percentage change.
5. **Inspect Residual Diagnostics:** Show the residual error plot and error distribution histogram.

---

### Step 7: Page 5 — Multi-Stock Comparison (`4_⚖️_Multi_Stock_Comparison.py`)
1. Select 4 stocks (e.g. Reliance, TCS, Infosys, HDFC Bank).
2. **Show Base-100 Normalized Chart:** Explain why comparing raw stock prices is misleading and why normalizing to base 100 reveals true relative performance.
3. **Inspect Pearson Correlation Heatmap:** Discuss diversification benefits when correlation between assets is low.

---

### Step 8: Page 6 — Data Management (`5_💾_Data_Management.py`)
1. **Show Storage Inventory:** Display cached raw CSVs, processed Parquet files, and sample datasets.
2. **Run Data Quality Audit:** Show the diagnostic report checking duplicate dates, null values, and price invariant logic.
3. **Show SQLite Audit Trail:** Inspect historical ingestion records stored in SQLite.

---

### Step 9: Conclude with Page 7 & Academic Limitations
1. Navigate to **About Project** to summarize technologies and architectural design.
2. Acknowledge academic limitations (Efficient Market Hypothesis, exogenous shocks, lack of order book depth) to demonstrate mature engineering mindset.
