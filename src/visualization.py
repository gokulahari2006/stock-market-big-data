"""Visualization module implementing professional Plotly charts with a dark financial theme.
Supports candlesticks, moving average overlays, indicators (RSI, MACD, Bollinger Bands),
residual diagnostics, multi-stock normalized returns, and benchmark visual comparisons.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import THEME_COLORS


def _apply_dark_theme(fig: go.Figure, title: str, height: int = 500) -> go.Figure:
    """Apply unified dark institutional finance styling to a Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "y": 0.95,
            "x": 0.05,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "color": THEME_COLORS["text_primary"]}
        },
        paper_bgcolor=THEME_COLORS["background"],
        plot_bgcolor=THEME_COLORS["card_background"],
        font={"color": THEME_COLORS["text_secondary"], "family": "Inter, Roboto, sans-serif"},
        height=height,
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        hovermode="x unified",
        legend={
            "bgcolor": "rgba(22, 27, 34, 0.8)",
            "bordercolor": THEME_COLORS["card_border"],
            "borderwidth": 1,
            "font": {"size": 11, "color": THEME_COLORS["text_primary"]}
        },
        xaxis={
            "gridcolor": THEME_COLORS["grid_color"],
            "showgrid": True,
            "zeroline": False,
            "linecolor": THEME_COLORS["card_border"]
        },
        yaxis={
            "gridcolor": THEME_COLORS["grid_color"],
            "showgrid": True,
            "zeroline": False,
            "linecolor": THEME_COLORS["card_border"]
        }
    )
    return fig


def create_price_line_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Historical closing price line chart with gradient area and volume subplot."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        subplot_titles=[f"{symbol} Closing Price History", "Daily Trading Volume"]
    )

    # Price Line
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["close"],
            name="Closing Price",
            line={"color": THEME_COLORS["accent_blue"], "width": 2},
            fill="tozeroy",
            fillcolor="rgba(41, 98, 255, 0.08)"
        ),
        row=1, col=1
    )

    # Volume Bar
    colors = [THEME_COLORS["accent_green"] if c >= o else THEME_COLORS["accent_red"]
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.8
        ),
        row=2, col=1
    )

    _apply_dark_theme(fig, f"{symbol} — Price & Volume Overview", height=550)
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def create_candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Interactive OHLC Candlestick chart with volume subplot."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22]
    )

    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color=THEME_COLORS["accent_green"],
            decreasing_line_color=THEME_COLORS["accent_red"]
        ),
        row=1, col=1
    )

    vol_colors = [THEME_COLORS["accent_green"] if c >= o else THEME_COLORS["accent_red"]
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.7
        ),
        row=2, col=1
    )

    _apply_dark_theme(fig, f"{symbol} — Candlestick & Volume Dynamics", height=600)
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def create_ma_overlay_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Closing price overlaid with multiple Simple and Exponential Moving Averages."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"],
        name="Close",
        line={"color": THEME_COLORS["text_primary"], "width": 1.5}
    ))

    colors = {
        "sma_20": "#00e5ff",
        "sma_50": "#ffab00",
        "sma_200": "#ff4081",
        "ema_12": "#00c076",
        "ema_26": "#9c27b0"
    }

    for col, color in colors.items():
        if col in df.columns and df[col].notna().sum() > 0:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col],
                name=col.upper(),
                line={"color": color, "width": 1.2, "dash": "solid" if "sma" in col else "dot"}
            ))

    return _apply_dark_theme(fig, f"{symbol} — Trend & Moving Average Overlays", height=500)


def create_bollinger_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Bollinger Bands (Upper, Middle, Lower) with volatility envelope shading."""
    fig = go.Figure()

    if "bollinger_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["bollinger_upper"],
            name="Upper Band (+2σ)",
            line={"color": "rgba(41, 98, 255, 0.4)", "width": 1}
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["bollinger_lower"],
            name="Lower Band (-2σ)",
            line={"color": "rgba(41, 98, 255, 0.4)", "width": 1},
            fill="tonexty",
            fillcolor="rgba(41, 98, 255, 0.05)"
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["bollinger_mid"],
            name="Middle Band (SMA 20)",
            line={"color": "#ffab00", "width": 1.2, "dash": "dash"}
        ))

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"],
        name="Close",
        line={"color": THEME_COLORS["text_primary"], "width": 1.5}
    ))

    return _apply_dark_theme(fig, f"{symbol} — Bollinger Bands (20-day, 2σ)", height=450)


def create_rsi_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Relative Strength Index (14) with overbought (70) and oversold (30) bands."""
    fig = go.Figure()
    if "rsi_14" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["rsi_14"],
            name="RSI (14)",
            line={"color": "#e040fb", "width": 1.8}
        ))

    # Add threshold lines
    fig.add_hline(y=70, line_dash="dash", line_color=THEME_COLORS["accent_red"], annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color=THEME_COLORS["accent_green"], annotation_text="Oversold (30)")
    fig.update_yaxes(range=[0, 100])

    return _apply_dark_theme(fig, f"{symbol} — Relative Strength Index (RSI 14)", height=320)


def create_macd_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """MACD Line (12, 26), 9-day Signal line, and Divergence Histogram."""
    fig = go.Figure()
    if "macd" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["macd"],
            name="MACD Line",
            line={"color": "#29b6f6", "width": 1.5}
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["macd_signal"],
            name="Signal Line",
            line={"color": "#ff7043", "width": 1.5}
        ))

        hist_colors = [THEME_COLORS["accent_green"] if val >= 0 else THEME_COLORS["accent_red"]
                       for val in df["macd_hist"]]
        fig.add_trace(go.Bar(
            x=df["date"], y=df["macd_hist"],
            name="MACD Histogram",
            marker_color=hist_colors
        ))

    return _apply_dark_theme(fig, f"{symbol} — MACD Oscillator (12, 26, 9)", height=340)


def create_volatility_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Rolling 20-day Annualized Volatility chart."""
    fig = go.Figure()
    col = "annualized_volatility_20" if "annualized_volatility_20" in df.columns else "volatility_20"
    if col in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[col],
            name="Annualized Volatility (%)",
            line={"color": THEME_COLORS["accent_amber"], "width": 2},
            fill="tozeroy",
            fillcolor="rgba(255, 171, 0, 0.08)"
        ))
    return _apply_dark_theme(fig, f"{symbol} — Rolling 20-Day Annualized Volatility (%)", height=350)


def create_prediction_comparison_chart(
    y_test: np.ndarray,
    predictions_dict: dict,
    dates: Optional[List[Any]] = None
) -> go.Figure:
    """Out-of-sample Actual vs Predicted comparison line chart across models."""
    fig = go.Figure()
    x_axis = dates if dates is not None and len(dates) == len(y_test) else list(range(len(y_test)))

    # Actual target
    fig.add_trace(go.Scatter(
        x=x_axis, y=y_test,
        name="Actual Next Close",
        line={"color": "#ffffff", "width": 2.5}
    ))

    colors = {
        "Persistence Baseline": "#8b949e",
        "Linear Regression": "#ffab00",
        "Random Forest": THEME_COLORS["accent_green"],
        "Gradient Boosting": "#00e5ff"
    }

    for model_name, preds in predictions_dict.items():
        if len(preds) == len(y_test):
            fig.add_trace(go.Scatter(
                x=x_axis, y=preds,
                name=f"{model_name} Predicted",
                line={"color": colors.get(model_name, "#2962ff"), "width": 1.8, "dash": "dash"}
            ))

    return _apply_dark_theme(fig, "Out-of-Sample Test Evaluation: Actual vs Predicted Prices", height=500)


def create_residuals_chart(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> go.Figure:
    """Residual error time-series and residual error distribution histogram."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=[f"{model_name} Prediction Residuals (₹)", "Error Distribution"]
    )

    fig.add_trace(
        go.Scatter(
            y=residuals,
            mode="lines+markers",
            marker={"size": 4, "color": THEME_COLORS["accent_blue"]},
            line={"width": 1},
            name="Residual (Actual - Pred)"
        ),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="solid", line_color=THEME_COLORS["accent_red"], row=1, col=1)

    fig.add_trace(
        go.Histogram(
            y=residuals,
            nbinsy=25,
            marker_color=THEME_COLORS["accent_green"],
            name="Residual Density"
        ),
        row=1, col=2
    )

    return _apply_dark_theme(fig, f"{model_name} — Residual Analysis Diagnostics", height=420)


def create_normalized_performance_chart(df_multi: pd.DataFrame) -> go.Figure:
    """Comparative normalized performance chart starting from base 100 for all tickers."""
    fig = go.Figure()
    symbols = df_multi["symbol"].unique()

    palette = ["#00c076", "#2962ff", "#ffab00", "#ff4081", "#00e5ff", "#e040fb", "#ff7043", "#76ff03"]

    for idx, sym in enumerate(symbols):
        sub = df_multi[df_multi["symbol"] == sym].sort_values(by="date")
        if not sub.empty:
            base_p = sub["close"].iloc[0]
            normalized_series = (sub["close"] / base_p) * 100.0
            fig.add_trace(go.Scatter(
                x=sub["date"],
                y=normalized_series,
                name=sym,
                line={"color": palette[idx % len(palette)], "width": 2}
            ))

    fig.add_hline(y=100, line_dash="dash", line_color="#8b949e", annotation_text="Baseline (100)")
    return _apply_dark_theme(fig, "Multi-Stock Normalized Performance Comparison (Base = 100)", height=500)


def create_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Interactive heatmap displaying correlation matrix between stock returns."""
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            colorscale="Viridis",
            zmin=-1.0,
            zmax=1.0,
            colorbar={"title": "Pearson ρ"}
        )
    )
    return _apply_dark_theme(fig, "Stock Returns Correlation Matrix (Pearson)", height=450)


def create_benchmark_chart(pandas_time: float, spark_time: Optional[float], record_count: int) -> go.Figure:
    """Execution time benchmark comparison between Pandas and PySpark."""
    categories = ["Pandas (In-Memory)", "Apache Spark (PySpark)"]
    times = [pandas_time, spark_time if spark_time else 0.0]
    colors = [THEME_COLORS["accent_blue"], THEME_COLORS["accent_amber"]]

    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=times,
            text=[f"{t:.4f}s" if t > 0 else "N/A" for t in times],
            textposition="auto",
            marker_color=colors,
            width=0.4
        )
    ])
    fig.update_layout(yaxis_title="Execution Duration (seconds)")
    return _apply_dark_theme(fig, f"Processing Engine Benchmark ({record_count:,} Records)", height=380)
