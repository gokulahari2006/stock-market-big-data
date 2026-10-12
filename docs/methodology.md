# Mathematical Methodology & Feature Engineering Formulation

## 1. Quantitative Financial Indicators

### 1.1 Daily Percentage Returns
The daily return $R_t$ at trading session $t$ measures the relative change in closing price relative to session $t-1$:
$$R_t = \left( \frac{C_t - C_{t-1}}{C_{t-1}} \right) \times 100\%$$
where $C_t$ represents the closing price at session $t$.

### 1.2 Simple Moving Average (SMA)
The unweighted mean of closing prices over a rolling sliding window of $k$ trading sessions:
$$\text{SMA}_k(t) = \frac{1}{k} \sum_{i=0}^{k-1} C_{t-i}$$
Supported window sizes: $k \in \{5, 10, 20, 50, 100, 200\}$.

### 1.3 Exponential Moving Average (EMA)
Assigns exponentially decreasing weights to older observations. Given smoothing factor $\alpha = \frac{2}{k + 1}$:
$$\text{EMA}_k(t) = \alpha \cdot C_t + (1 - \alpha) \cdot \text{EMA}_k(t-1)$$
Supported windows: Fast ($k=12$), Slow ($k=26$).

### 1.4 Rolling Annualized Volatility
Sample standard deviation of daily percentage returns over a 20-day window, annualized by scaling with the square root of 252 annual trading sessions:
$$\sigma_{20}(t) = \sqrt{\frac{1}{19} \sum_{i=0}^{19} (R_{t-i} - \bar{R}_t)^2}$$
$$\sigma_{\text{annual}}(t) = \sigma_{20}(t) \times \sqrt{252}$$

### 1.5 Relative Strength Index (RSI - 14 Periods)
Measures the velocity and magnitude of directional price movements:
$$U_t = \max(C_t - C_{t-1}, 0), \quad D_t = \max(C_{t-1} - C_t, 0)$$
Using Wilder's exponential smoothing with $\alpha = \frac{1}{14}$:
$$\text{AvgGain}_t = \frac{1}{14} U_t + \frac{13}{14} \text{AvgGain}_{t-1}$$
$$\text{AvgLoss}_t = \frac{1}{14} D_t + \frac{13}{14} \text{AvgLoss}_{t-1}$$
$$\text{RS}_t = \frac{\text{AvgGain}_t}{\text{AvgLoss}_t + \epsilon}, \quad \text{RSI}_t = 100 - \left( \frac{100}{1 + \text{RS}_t} \right)$$

### 1.6 Moving Average Convergence Divergence (MACD)
$$\text{MACD}_t = \text{EMA}_{12}(t) - \text{EMA}_{26}(t)$$
$$\text{Signal}_t = \text{EMA}_9(\text{MACD}_t)$$
$$\text{Histogram}_t = \text{MACD}_t - \text{Signal}_t$$

### 1.7 Bollinger Bands
$$\text{Upper}_t = \text{SMA}_{20}(t) + 2 \cdot \sigma_{20}(t)$$
$$\text{Middle}_t = \text{SMA}_{20}(t)$$
$$\text{Lower}_t = \text{SMA}_{20}(t) - 2 \cdot \sigma_{20}(t)$$

---

## 2. Feature Engineering & Target Formulation

### 2.1 Target Variable Definition
The prediction target is strictly defined as the closing price of the next trading session $t+1$:
$$y_t = C_{t+1}$$

### 2.2 Feature Matrix Formulation ($X_t$)
Every observation row $t$ consists exclusively of information known on or before the close of session $t$:
$$X_t = \Big[ C_t, C_{t-1}, C_{t-2}, C_{t-4}, C_{t-9}, C_{t-19}, R_t, R_{t-4}, R_{t-9}, V_t, V_{t-4}, \text{SMA}_5(t), \text{SMA}_{10}(t), \text{SMA}_{20}(t), \text{EMA}_{12}(t), \text{EMA}_{26}(t), \sigma_{20}(t), \text{RSI}_{14}(t), \text{MACD}_t, \text{Signal}_t \Big]$$

### 2.3 Strict Data Leakage Prevention Proof
1. **Temporal Causality:** No feature in $X_t$ references price, volume, or indicator data from $t+k$ where $k \ge 1$.
2. **Train/Test Splitting:** If total observations are $T$, the dataset is split chronologically:
   $$\mathcal{D}_{\text{train}} = \{(X_t, y_t)\}_{t=1}^{\lfloor 0.8T \rfloor}, \quad \mathcal{D}_{\text{test}} = \{(X_t, y_t)\}_{t=\lfloor 0.8T \rfloor + 1}^{T-1}$$
3. **Normalization Boundary:** Feature normalization $\mu, \sigma$ are fit strictly on $\mathcal{D}_{\text{train}}$:
   $$X_{\text{scaled}} = \frac{X - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
4. **Next-Session Inference:** For forecasting session $T+1$, feature vector $X_T$ is extracted from the most recent historical row, where target $y_T = C_{T+1}$ is currently unknown.

---

## 3. Supervised Machine Learning Models

### 3.1 Persistence Baseline
A naive theoretical baseline assuming the stock market follows a random walk:
$$\hat{y}_{t+1} = C_t$$
Any viable predictive model must systematically outperform this baseline.

### 3.2 Linear Regression
Models closing price as a linear combination of input features:
$$\hat{y}_{t+1} = \beta_0 + \sum_{j=1}^p \beta_j X_{t,j}$$
Fitted via Ordinary Least Squares (OLS) minimizing the sum of squared residuals.

### 3.3 Random Forest Regression
Ensemble of $B = 100$ decorrelated decision regression trees:
$$\hat{y}_{t+1} = \frac{1}{B} \sum_{b=1}^B T_b(X_t)$$
Hyperparameters: `max_depth=10`, `min_samples_split=5`, `random_state=42`.

### 3.4 Gradient Boosting Regression
Sequential additive modeling where each tree fits the negative gradient (pseudo-residuals) of the loss function:
$$F_m(X_t) = F_{m-1}(X_t) + \eta \cdot h_m(X_t)$$

---

## 4. Evaluation Metrics

### 4.1 Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

### 4.2 Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

### 4.3 Mean Absolute Percentage Error (MAPE)
$$\text{MAPE} = \frac{1}{N} \sum_{i=1}^N \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%$$

### 4.4 Directional Accuracy
$$\text{Directional Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}_{\text{sign}(\hat{y}_i - C_{i,\text{prev}}) == \text{sign}(y_i - C_{i,\text{prev}})} \times 100\%$$
