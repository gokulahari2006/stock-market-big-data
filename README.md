# Stock Market Data Analysis and Prediction Using Big Data Analytics and Machine Learning

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5-orange.svg)](https://spark.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An institutional-grade, end-to-end academic major project demonstrating historical financial data ingestion, distributed big data analytics with **Apache Spark (PySpark)**, leak-free feature engineering, and supervised machine learning forecasting on Indian equity markets (**NSE/Yahoo Finance**).

---

## 1. Abstract
This project bridges the gap between big data distributed processing and predictive machine learning in quantitative finance. The platform ingests multi-year historical stock market data for major Indian equities, performs schema validation, executes distributed window and aggregation transformations using PySpark DataFrames, extracts 20 causal lag features with zero lookahead bias, and trains supervised regression models (**Random Forest Regressor**, **Linear Regression**, and **Gradient Boosting**) benchmarked against a naive **Persistence Baseline**. The system is packaged within an institutional dark financial dashboard built using **Streamlit** and **Plotly**, with persistent **SQLite** audit logs and **Apache Parquet** columnar datasets.

---

## 2. Key Features
- **Multi-Tiered Data Collection:** Yahoo Finance live integration (`yfinance`) with automatic local caching and verified offline sample fixtures for seamless offline demonstration.
- **Data Validation & Quality Audits:** Real-time checking of financial invariants ($H \ge L, O > 0, C > 0, V \ge 0$), duplicate detection, and automated quality reports.
- **Distributed Big Data Engine (PySpark):** Real Apache Spark DataFrame transformations, window lag operations, rolling moving averages, multi-symbol aggregations, and Parquet exports.
- **Pandas vs PySpark Benchmark:** Empirical, transparent runtime comparison measuring execution duration across scalable record counts.
- **Quantitative Technical Indicators:** SMA (5, 10, 20, 50, 100, 200), EMA (12, 26), Bollinger Bands, RSI 14, MACD, and 20-day annualized volatility.
- **Zero Target Leakage ML Pipeline:** Chronological 80/20 train/test splitting; all features strictly lag observations up to session $t$ to forecast session $t+1$.
- **Model Evaluation Suite:** MAE, RMSE, MAPE, and Directional Accuracy metrics evaluated on held-out test data.
- **Next-Session Inference:** Point estimate forecasting for the upcoming trading session with absolute and percentage change indicators.
- **Multi-Stock Comparison:** Base-100 normalized performance charts and Pearson return correlation heatmaps.
- **Relational Metadata & Audit:** SQLite tracking of ingestion jobs, model evaluations, and forecast history.

---

## 3. Technology Stack

| Domain | Technology / Library | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.11 / 3.12 | Core programming runtime |
| **Big Data Engine** | Apache Spark (PySpark) | 3.5+ | Distributed DataFrame transformations & window functions |
| **Web Dashboard** | Streamlit | 1.32+ | Interactive multi-page reactive user interface |
| **Visualizations** | Plotly | 5.19+ | Institutional dark candlestick & indicator charts |
| **Data Ingestion** | yfinance | 0.2.36+ | Historical market data collection |
| **Data Processing** | Pandas & NumPy | 2.2+ / 1.26+ | Local in-memory manipulation & numerical calculations |
| **Storage Formats** | Apache Parquet & CSV | PyArrow 15+ | Columnar high-performance analytical storage |
| **Machine Learning**| Scikit-Learn & Joblib | 1.4+ / 1.3+ | Random Forest, Linear Regression, serialization |
| **Audit Database** | SQLite | 3.x (Built-in) | Relational ingestion and evaluation logging |
| **Testing** | Pytest | 8.0+ | Automated deterministic unit testing |

---

## 4. System Architecture

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

---

## 5. Folder Structure

```
stock-market-big-data/
|-- app.py                                # Main Streamlit dashboard (Market Overview)
|-- requirements.txt                      # Project dependency specification
|-- README.md                             # Comprehensive technical manual
|-- .gitignore                            # Version control exclusion rules
|-- pytest.ini                            # Pytest configuration file
|
|-- pages/                                # Multi-page Streamlit dashboards
|   |-- 1_📈_Stock_Explorer.py            # Candlestick, moving averages, RSI, MACD
|   |-- 2_⚡_Big_Data_Analytics.py        # PySpark DataFrame pipeline & Pandas benchmark
|   |-- 3_🤖_Stock_Prediction.py          # ML model training & next-session forecasting
|   |-- 4_⚖️_Multi_Stock_Comparison.py    # Base-100 normalized returns & correlation heatmap
|   |-- 5_💾_Data_Management.py           # Storage inventory, quality audit & SQLite logs
|   |-- 6_📊_Model_Evaluation.py          # Detailed residual diagnostics & metric breakdown
|   |-- 7_ℹ️_About_Project.py             # Architecture, technology stack & viva questions
|
|-- src/                                  # Core business logic & analytics packages
|   |-- __init__.py
|   |-- config.py                         # System paths, constants, ticker dictionary
|   |-- utils.py                          # High-resolution execution timer & formatters
|   |-- database.py                       # SQLite manager for audit & evaluation logs
|   |-- storage.py                        # File storage manager for CSV, Parquet, models
|   |-- data_validation.py                # Schema & financial invariant validation
|   |-- preprocessing.py                  # Deduplication, sorting, and cleaning pipeline
|   |-- data_collection.py                # yfinance API fetcher & offline sample fallback
|   |-- financial_analytics.py            # Indicators: SMA, EMA, Volatility, RSI, MACD
|   |-- feature_engineering.py            # 20 causal lag features & target formulation
|   |-- prediction_model.py               # LR, Random Forest, Gradient Boosting, baseline
|   |-- model_evaluation.py               # MAE, RMSE, MAPE, Directional Accuracy calculators
|   |-- spark_processor.py                # PySpark pipeline, window functions & benchmark
|   |-- visualization.py                  # Plotly dark financial chart generator
|
|-- data/
|   |-- raw/                              # Cached raw downloads
|   |-- processed/                        # Cleaned CSV & Parquet outputs
|   |-- sample/                           # Pre-generated verified offline fixtures
|
|-- models/                               # Serialized Joblib model artifacts
|-- database/                             # SQLite relational database (stock_analytics.db)
|
|-- tests/                                # Automated unit test suite
|   |-- test_data_validation.py
|   |-- test_preprocessing.py
|   |-- test_financial_analytics.py
|   |-- test_feature_engineering.py
|   |-- test_prediction_model.py
|   |-- test_database.py
|   |-- test_spark_processor.py
|
|-- docs/                                 # Academic documentation & viva preparation
|   |-- architecture.md
|   |-- methodology.md
|   |-- viva_questions.md
|   |-- demonstration_guide.md
|   |-- project_report.md
```

---

## 6. Installation & Environment Setup

### System Prerequisites
- **Python:** Python 3.11 or 3.12 (64-bit recommended)
- **Java Runtime (for Apache Spark):** Java 8, 11, or 17 (Java 8 is pre-installed in this environment). Verify via `java -version`.

### Step 1: Create Virtual Environment
#### Windows (PowerShell):
```powershell
# Navigate into the project folder
cd stock-market-big-data

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

#### Linux / macOS (Bash):
```bash
cd stock-market-big-data
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. Execution Instructions

### Running the Web Application Locally
Launch the interactive Streamlit terminal:
```powershell
# Windows
python -m streamlit run app.py

# Linux / macOS
streamlit run app.py
```
The browser will automatically open at:
```text
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### Running Automated Tests
Run the deterministic unit test suite:
```bash
pytest -v
```

---

## 8. Online Cloud Deployment Guide

The application is engineered to be portable and deployable across all major cloud hosting platforms without code modifications.

### Deployment Option A: Streamlit Community Cloud (Recommended & Free)
1. **Push Code to GitHub:**
   Commit the project repository to your GitHub account (public or private):
   ```bash
   git init
   git add .
   git commit -m "Deploy Stock Market Big Data Analytics Platform"
   git remote add origin https://github.com/<your-username>/stock-market-big-data.git
   git push -u origin main
   ```
2. **Deploy on Streamlit Cloud:**
   - Log into [share.streamlit.io](https://share.streamlit.io/).
   - Click **"New app"**.
   - Select your repository (`stock-market-big-data`), branch (`main`), and set **Main file path** to `app.py`.
   - Click **"Deploy"**.
3. **Automated Java Provisioning:**
   - Streamlit Cloud detects [`packages.txt`](packages.txt) and automatically installs `default-jre-headless` via Debian `apt-get`.
   - If Java is ever absent or constrained by memory limits, the application activates its built-in graceful fallback to the local Pandas engine with zero crashes.

---

### Deployment Option B: Docker Container (AWS, GCP, Azure, DigitalOcean)
The project includes a multi-stage, production-ready [`Dockerfile`](Dockerfile) with Java pre-installed.

1. **Build the Docker Image:**
   ```bash
   docker build -t stock-market-analytics:latest .
   ```
2. **Run the Container Locally:**
   ```bash
   docker run -d -p 8501:8501 --name stock-analytics stock-market-analytics:latest
   ```
   Open `http://localhost:8501` in your browser.
3. **Deploy to Cloud Containers:**
   - **Google Cloud Run:**
     ```bash
     gcloud builds submit --tag gcr.io/<PROJECT-ID>/stock-analytics
     gcloud run deploy stock-analytics --image gcr.io/<PROJECT-ID>/stock-analytics --port 8501 --allow-unauthenticated
     ```
   - **AWS App Runner / ECS:** Push the Docker image to AWS ECR and deploy as a web service targeting container port `8501`.

---

### Deployment Option C: Render / Railway / PaaS
1. Connect your GitHub repository to Render or Railway.
2. Configure service settings:
   - **Environment:** Python 3.11 / 3.12 or Docker
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
3. If deploying via Native Python runtime on Render, add an environment build script or install Java via Apt buildpack if PySpark cluster execution is required.

---

### Deployment Option D: Linux Virtual Machine / VPS (Ubuntu / Debian)
1. **Provision Dependencies & Java Runtime:**
   ```bash
   sudo apt update && sudo apt install -y python3-pip python3-venv default-jre-headless git
   ```
2. **Clone & Setup Environment:**
   ```bash
   git clone https://github.com/<your-username>/stock-market-big-data.git
   cd stock-market-big-data
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Run with Systemd Service (`/etc/systemd/system/stockapp.service`):**
   ```ini
   [Unit]
   Description=Stock Market Analytics Streamlit App
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/stock-market-big-data
   ExecStart=/home/ubuntu/stock-market-big-data/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```
   Enable and start the service:
   ```bash
   sudo systemctl enable stockapp
   sudo systemctl start stockapp
   ```

---

## 9. Java & PySpark Compatibility Matrix

| PySpark Version | Compatible Java Runtimes | Recommended JRE Package | Notes |
| :--- | :--- | :--- | :--- |
| **PySpark 3.5.x (LTS)** | Java 8, Java 11, Java 17 | `default-jre-headless` | Most stable and recommended for cloud production |
| **PySpark 4.0.x** | Java 17, Java 21 | `openjdk-17-jre-headless` | Requires bytecode class file 61.0+ |

### Fallback Guarantee
If the cloud host does not provide a Java runtime, the system's `check_java_compatibility()` routine in [`src/spark_processor.py`](src/spark_processor.py) intercepts the initialization in milliseconds, displays a clear notification banner, and falls back to vectorised Pandas processing. The application will never crash or hang on cloud initialization.

---

## 10. Cloud Storage & Ephemeral File Safety
- **Dynamic Directory Provisioning:** All vital storage directories (`data/raw/`, `data/processed/`, `models/`, `database/`) are automatically verified and created at startup via [`src/config.py`](src/config.py).
- **SQLite Database Auto-Migration:** The SQLite database (`database/stock_analytics.db`) executes `CREATE TABLE IF NOT EXISTS` on initial boot with zero manual database administration.
---

## 11. Machine Learning Workflow & Leakage Prevention

1. **Target Definition:** Target is strictly the closing price at session $t+1$: $y_t = C_{t+1}$.
2. **Causal Lag Features:** Features at index $t$ are constructed strictly using observations known on or before day $t$.
3. **Chronological Splitting:** The dataset is split into earliest 80% (Train) and latest 20% (Test) without shuffling.
4. **StandardScaler Bounds:** Fitted strictly on the training set to prevent future distribution statistics from leaking into features.
5. **Evaluation Against Baseline:** Models are evaluated on out-of-sample test data against the persistence baseline ($\hat{y}_{t+1} = C_t$).

---

## 12. Academic Viva Questions
Comprehensive answers to over 30 common examiner questions are documented in [`docs/viva_questions.md`](docs/viva_questions.md).

---

## 13. Educational Disclaimer
**DISCLAIMER:** This application is developed strictly for educational, academic, and research purposes as part of a college major project. Stock market predictions are statistical approximations based on historical data and do **NOT** guarantee future performance or market profits. They must not be construed as investment, financial, or trading advice. `yfinance` is an unofficial API subject to Yahoo Finance terms of service.
