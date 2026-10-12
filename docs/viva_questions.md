# Comprehensive College Viva Examination Questions & Answers

This document provides rigorous, technical answers for college project viva-voce evaluations for the project:
**"Stock Market Data Analysis and Prediction Using Big Data Analytics and Machine Learning"**.

---

### Q1: What is Big Data Analytics?
**Answer:** Big Data Analytics refers to the systematic extraction of actionable insights, statistical patterns, and hidden correlations from complex, high-volume, high-velocity datasets that exceed the storage, memory, or processing capacity of conventional single-node relational database systems. It leverages distributed computing architectures (e.g. Apache Spark, Hadoop) to divide computations across partitions and nodes.

### Q2: Why is this considered a Big Data project?
**Answer:** In capital markets, financial tick data and multi-asset histories scale exponentially. A production market surveillance system must continuously ingest, validate, aggregate, and process multi-gigabyte or terabyte streams across thousands of securities. Our project implements this exact big data paradigm: PySpark DataFrames, explicit `StructType` schemas, distributed window operations (`Window.partitionBy().orderBy()`), and columnar Apache Parquet storage, allowing linear horizontal scalability.

### Q3: What is Apache Spark?
**Answer:** Apache Spark is an open-source, distributed general-purpose cluster-computing framework. It provides an optimized execution engine with in-memory caching and resilient distributed datasets (RDDs), making data processing significantly faster than legacy disk-bound MapReduce.

### Q4: What is PySpark?
**Answer:** PySpark is the Python API for Apache Spark. It exposes Spark's distributed computing capabilities (Spark SQL, DataFrame API, Streaming, MLlib) using idiomatic Python syntax, communicating with the underlying JVM via Py4J gateway sockets.

### Q5: Why is Pandas used alongside PySpark in this application?
**Answer:** Pandas excels at fast, in-memory exploratory data analysis, interactive manipulation of single-asset time series, and immediate feeding into Scikit-Learn models on a single workstation. PySpark is utilized for the distributed multi-stock aggregation pipeline, window functions, and Parquet persistence.

### Q6: What is the fundamental difference between Pandas and PySpark?
**Answer:** 
- **Memory Model:** Pandas operates strictly in a single process's RAM on a single CPU core/machine; exceeding available RAM results in an `OutOfMemoryError`. PySpark partitions datasets across multiple executor JVM processes and worker nodes.
- **Evaluation:** Pandas uses eager evaluation (every operation runs immediately). PySpark uses lazy evaluation (builds a Directed Acyclic Graph / DAG of transformations and optimizes it using the Catalyst Optimizer before executing actions like `count()` or `collect()`).

### Q7: Why can Pandas be faster than PySpark on small datasets?
**Answer:** PySpark incurs substantial fixed overhead: launching JVM processes, socket serialization/deserialization between Python and Java (Py4J), and query planning. On small datasets (< 100,000 rows), this fixed overhead exceeds the actual computation time. Pandas runs compiled C code directly in memory with zero IPC overhead.

### Q8: What is stock market OHLCV data?
**Answer:** OHLCV stands for Open, High, Low, Close, and Volume:
- **Open:** The execution price of the first trade in a trading session.
- **High:** The highest transaction price reached during the session.
- **Low:** The lowest transaction price reached during the session.
- **Close:** The cash settlement price of the final trade of the session.
- **Volume:** The aggregate number of shares transacted during the session.

### Q9: What is a Moving Average and why is it used?
**Answer:** A moving average smooths high-frequency noise from price series to identify underlying trend trajectories.
- **Simple Moving Average (SMA):** An arithmetic mean over a sliding window of $k$ sessions.
- **Exponential Moving Average (EMA):** Places exponential weight on the most recent trading sessions using smoothing constant $\alpha = \frac{2}{k+1}$, reacting faster to recent price inflections.

### Q10: What is Historical Volatility?
**Answer:** Volatility quantifies the dispersion of price returns over time. In our project, rolling 20-day volatility is the sample standard deviation of daily percentage returns, annualized by multiplying by $\sqrt{252}$ (the number of trading days in a typical calendar year).

### Q11: What is Feature Engineering?
**Answer:** Feature engineering is the domain-driven transformation of raw data into informative predictor variables that expose patterns to machine learning algorithms. In our project, we generate 20 causal features: lagged closing prices (lags 1, 2, 3, 5, 10, 20), lagged returns, volume lags, moving averages (SMA 5, 10, 20), EMAs (12, 26), 20-day volatility, RSI, and MACD.

### Q12: What is Supervised Machine Learning in the context of this project?
**Answer:** Supervised learning learns a mapping function $f: X \to y$ from labeled historical training pairs. Here, input vector $X_t$ consists of indicators up to session $t$, and the continuous target $y_t$ is the closing price of the next trading session $C_{t+1}$ (a regression problem).

### Q13: Why is Linear Regression used as a baseline?
**Answer:** Linear Regression provides an interpretable parametric baseline: $\hat{y} = \beta_0 + \sum \beta_j X_j$. It verifies whether a simple linear combination of past prices and indicators has predictive power before introducing complex non-linear ensemble models.

### Q14: What is Random Forest Regression?
**Answer:** Random Forest is an ensemble learning method based on bootstrap aggregation (bagging) of decorrelated decision trees. It builds $B$ decision trees on randomized bootstrap sub-samples and randomly selects a subset of features at each node split. The final prediction is the arithmetic average of all individual tree predictions, substantially reducing variance and preventing overfitting.

### Q15: What is Chronological Train/Test Splitting and why is it mandatory for time-series?
**Answer:** In time-series analysis, observations possess temporal dependency (autocorrelation). Randomly shuffling rows (like standard K-Fold cross-validation) leaks future information into past predictions. We strictly split chronologically: the earliest 80% for training and the final 20% for testing.

### Q16: What is Data Leakage / Lookahead Bias?
**Answer:** Data leakage occurs when information from the target or from the future is inadvertently included in model features during training. In our pipeline, every feature at index $t$ uses data strictly available on or before day $t$. Future indicators and future prices are strictly excluded.

### Q17: What is Mean Absolute Error (MAE)?
**Answer:** MAE is the average magnitude of absolute errors:
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
It represents the expected error in the actual currency unit (₹) without heavily penalizing large outliers.

### Q18: What is Root Mean Squared Error (RMSE)?
**Answer:** RMSE is the square root of the mean squared prediction errors:
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
Because errors are squared, RMSE penalizes large prediction errors more severely than MAE.

### Q19: What is Mean Absolute Percentage Error (MAPE)?
**Answer:** MAPE expresses forecast errors as a percentage of actual values:
$$\text{MAPE} = \frac{1}{N} \sum_{i=1}^N \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%$$
It provides a scale-independent metric suitable for comparing accuracy across different stock price regimes.

### Q20: What is Directional Accuracy and why is it important in finance?
**Answer:** Directional Accuracy measures the proportion of times the model correctly predicts whether the price will move Up or Down relative to the latest close:
$$\text{Directional Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}_{\text{sign}(\hat{y}_{t+1} - C_t) == \text{sign}(y_{t+1} - C_t)} \times 100\%$$
In quantitative trading, knowing the correct direction is often more critical for profitability than the exact price level.

### Q21: What is the Persistence Baseline?
**Answer:** The persistence model predicts that tomorrow's closing price will equal today's closing price: $\hat{y}_{t+1} = C_t$. Because financial markets resemble a martingale / random walk, the persistence baseline often achieves deceptively low RMSE. Any viable machine learning model must be evaluated against this baseline.

### Q22: How is the next-session forecast generated?
**Answer:** The feature engineer extracts the very last available row from history (session $T$). Its features (lags, moving averages, indicators) are passed to the trained model pipeline to output the forecasted point estimate $\hat{C}_{T+1}$ for the upcoming unobserved trading day.

### Q23: What are the fundamental limitations of predicting stock prices?
**Answer:**
1. **Efficient Market Hypothesis (EMH):** New information is rapidly reflected in prices.
2. **Exogenous Events:** Macroeconomic announcements, geopolitical conflicts, and quarterly earnings surprises cannot be inferred from historical price charts alone.
3. **Non-stationarity:** Financial time-series statistical distributions shift over time (regime changes).

### Q24: Why is Apache Parquet used for dataset storage?
**Answer:** Apache Parquet is an open-source, columnar storage format optimized for big data analytics. Unlike CSV (which is row-oriented text), Parquet uses dictionary encoding, run-length compression, and column-wise statistics, drastically reducing disk I/O and query scanning times.

### Q25: What is SQLite and how is it used in this project?
**Answer:** SQLite is a serverless, self-contained, embedded SQL relational database engine. In this project, it maintains structured audit logs of data ingestion, historical model evaluation metrics, and forecast records in `database/stock_analytics.db`.

### Q26: What is Streamlit and why was it chosen?
**Answer:** Streamlit is a modern Python framework for building interactive, reactive data applications. It allows rapid translation of analytical models and data pipelines into production-grade multi-page web dashboards with state management.

### Q27: What is Plotly?
**Answer:** Plotly is an interactive, browser-based graphing library. It renders responsive SVG and WebGL visualizations with native support for tooltips, zoom, pan, and range sliders, enabling institutional-grade financial charts.

### Q28: How is the application tested?
**Answer:** The project uses **Pytest** with synthetic, deterministic fixtures across seven specialized test modules (`test_data_validation`, `test_preprocessing`, `test_financial_analytics`, `test_feature_engineering`, `test_prediction_model`, `test_database`, `test_spark_processor`). Unit tests run offline without relying on external internet connectivity.

### Q29: How is execution time accurately measured?
**Answer:** The project uses Python's high-resolution `time.perf_counter()` context manager (`execution_timer`), which queries the hardware performance counter to measure elapsed execution duration down to sub-millisecond precision.

### Q30: What happens if the live yfinance API is unavailable or rate-limited?
**Answer:** The data collector uses a multi-tiered failover strategy: it first checks the local raw CSV cache; if absent and the network fails, it seamlessly falls back to pre-generated, verified historical sample datasets stored in `data/sample/`, notifying the user with an explicit demonstration badge.

### Q31: What are the primary future enhancements for this project?
**Answer:**
1. Integrating Apache Kafka for streaming real-time order-book ticks into Spark Structured Streaming.
2. Incorporating NLP sentiment analysis on financial news headlines using FinBERT.
3. Implementing Deep Learning sequence architectures (LSTM and Temporal Fusion Transformers).
4. Automated portfolio optimization (Markowitz Mean-Variance) and backtesting execution engines with slippage.
