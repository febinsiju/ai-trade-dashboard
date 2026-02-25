import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide", page_title="AI Trading Intelligence")

# =============================
# SIDEBAR
# =============================
st.sidebar.title("🤖 AI Trading Engine")

symbol = st.sidebar.selectbox(
    "Select Asset",
    ["BTC-USD", "ETH-USD", "AAPL", "TSLA"]
)

period = st.sidebar.selectbox(
    "Backtest Period",
    ["6mo", "1y", "2y", "5y"]
)

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data(symbol, period):
    return yf.download(symbol, period=period, progress=False)

data = load_data(symbol, period)

if data.empty:
    st.error("No data available")
    st.stop()

# =============================
# FEATURE ENGINEERING
# =============================
data["MA10"] = data["Close"].rolling(10).mean()
data["MA50"] = data["Close"].rolling(50).mean()
data["Return"] = data["Close"].pct_change()
data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

data = data.dropna()

# =============================
# TRAIN MODEL
# =============================
X = data[["MA10", "MA50"]]
y = data["Target"]

split = int(len(data) * 0.7)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

data_test = data.iloc[split:].copy()
data_test["Prediction"] = predictions

# =============================
# BACKTEST STRATEGY
# =============================
data_test["Strategy_Return"] = data_test["Return"] * data_test["Prediction"]
data_test["Cumulative_Strategy"] = (1 + data_test["Strategy_Return"]).cumprod()
data_test["Cumulative_Market"] = (1 + data_test["Return"]).cumprod()

total_strategy_return = round(
    (data_test["Cumulative_Strategy"].iloc[-1] - 1) * 100, 2
)

total_market_return = round(
    (data_test["Cumulative_Market"].iloc[-1] - 1) * 100, 2
)

win_rate = round(
    (data_test["Prediction"] == data_test["Target"]).mean() * 100, 2
)

# =============================
# MAIN LAYOUT
# =============================
st.title("📊 AI Trading Intelligence Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("AI Win Rate", f"{win_rate}%")
col2.metric("Strategy Return", f"{total_strategy_return}%")
col3.metric("Buy & Hold Return", f"{total_market_return}%")

# =============================
# PRICE CHART WITH SIGNALS
# =============================
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data_test.index,
    open=data_test["Open"],
    high=data_test["High"],
    low=data_test["Low"],
    close=data_test["Close"],
    name="Price"
))

# BUY markers
buy_signals = data_test[data_test["Prediction"] == 1]

fig.add_trace(go.Scatter(
    x=buy_signals.index,
    y=buy_signals["Close"],
    mode="markers",
    marker=dict(size=8),
    name="BUY Signal"
))

fig.update_layout(
    template="plotly_dark",
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# =============================
# EQUITY CURVE
# =============================
st.subheader("📈 Strategy vs Market Performance")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=data_test.index,
    y=data_test["Cumulative_Strategy"],
    name="AI Strategy"
))

fig2.add_trace(go.Scatter(
    x=data_test.index,
    y=data_test["Cumulative_Market"],
    name="Buy & Hold"
))

fig2.update_layout(
    template="plotly_dark",
    height=400
)

st.plotly_chart(fig2, use_container_width=True)
