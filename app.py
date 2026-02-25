import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(
    page_title="AI Trading Terminal Pro",
    layout="wide",
    page_icon="📈"
)

# ==================================
# PROFESSIONAL DARK THEME
# ==================================
st.markdown("""
<style>
body { background-color: #0E1117; }
.main { background-color: #0E1117; }
.block-container { padding-top: 1rem; }
h1 { color: #00FFA3; }

.signal-buy {
    background-color: rgba(0,255,163,0.15);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

.signal-sell {
    background-color: rgba(255,60,60,0.15);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ==================================
# SIDEBAR CONTROLS
# ==================================
st.sidebar.title("⚙️ Trading Controls")

symbol = st.sidebar.text_input("Asset Symbol", "BTC-USD")
period = st.sidebar.selectbox("Historical Period", ["6mo", "1y", "2y", "5y"])
backtest_toggle = st.sidebar.checkbox("Enable Backtesting", value=True)

# ==================================
# LOAD DATA
# ==================================
@st.cache_data
def load_data(symbol, period):
    return yf.download(symbol, period=period, progress=False)

data = load_data(symbol, period)

st.title("📊 AI Trading Intelligence Terminal")

if data.empty:
    st.error("No data found.")
    st.stop()

# ==================================
# INDICATORS
# ==================================
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()

# RSI
delta = data["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
data["RSI"] = 100 - (100 / (1 + rs))

# MACD
data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"] = data["EMA12"] - data["EMA26"]
data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
data = data.dropna()

# ==================================
# MODEL TRAINING
# ==================================
features = ["MA20", "MA50", "RSI", "MACD"]
X = data[features]
y = data["Target"]

split = int(len(data) * 0.8)

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = round(accuracy_score(y_test, predictions) * 100, 2)

latest = X.iloc[-1:]
prediction = model.predict(latest)
prob = model.predict_proba(latest)
confidence = round(np.max(prob) * 100, 2)

# ==================================
# METRICS PANEL
# ==================================
col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", f"{accuracy}%")
col2.metric("Confidence", f"{confidence}%")

current_price = round(float(data["Close"].iloc[-1]), 2)
col3.metric("Current Price", f"${current_price}")

# ==================================
# MAIN CHART
# ==================================
st.subheader("Price Chart")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Price"
))

fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20"))
fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50"))

fig.update_layout(
    template="plotly_dark",
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# ==================================
# RSI & MACD PANELS
# ==================================
col_rsi, col_macd = st.columns(2)

with col_rsi:
    st.subheader("RSI Indicator")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI"))
    fig_rsi.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_rsi, use_container_width=True)

with col_macd:
    st.subheader("MACD Indicator")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD"))
    fig_macd.add_trace(go.Scatter(x=data.index, y=data["Signal_Line"], name="Signal"))
    fig_macd.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_macd, use_container_width=True)

# ==================================
# AI SIGNAL PANEL
# ==================================
st.subheader("AI Trade Signal")

if prediction[0] == 1:
    st.markdown('<div class="signal-buy"><h2>📈 BUY</h2></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="signal-sell"><h2>📉 SELL</h2></div>', unsafe_allow_html=True)

# ==================================
# BACKTEST SECTION
# ==================================
if backtest_toggle:

    st.subheader("Strategy Backtest")

    data_test = data.iloc[split:].copy()
    data_test["Prediction"] = predictions
    data_test["Return"] = data_test["Close"].pct_change()
    data_test["Strategy"] = data_test["Return"] * data_test["Prediction"]

    data_test["Cumulative_Market"] = (1 + data_test["Return"]).cumprod()
    data_test["Cumulative_Strategy"] = (1 + data_test["Strategy"]).cumprod()

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=data_test.index, y=data_test["Cumulative_Market"], name="Buy & Hold"))
    fig_bt.add_trace(go.Scatter(x=data_test.index, y=data_test["Cumulative_Strategy"], name="AI Strategy"))
    fig_bt.update_layout(template="plotly_dark", height=400)

    st.plotly_chart(fig_bt, use_container_width=True)
