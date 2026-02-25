import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide", page_title="AI Trading Research Terminal")

# ==============================
# PROFESSIONAL DARK STYLE
# ==============================
st.markdown("""
<style>
body { background-color: #0E1117; }
.main { background-color: #0E1117; }
.block-container { padding-top: 1rem; }

.control-box {
    background-color: #141A2A;
    padding: 15px;
    border-radius: 12px;
}

.signal-buy {
    background-color: rgba(0,255,163,0.15);
    padding: 15px;
    border-radius: 12px;
    text-align:center;
}

.signal-sell {
    background-color: rgba(255,60,60,0.15);
    padding: 15px;
    border-radius: 12px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 AI Trading Research Terminal")

# ==============================
# TOP CENTER CONTROLS
# ==============================
st.markdown('<div class="control-box">', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)

symbol = c1.text_input("Symbol", "BTC-USD")
period = c2.selectbox("Period", ["6mo", "1y", "2y", "5y"])
model_type = c3.selectbox("Model", ["Random Forest"])
risk_pct = c4.slider("Risk % per Trade", 1, 10, 2)
backtest_toggle = c5.checkbox("Backtest", True)
run_button = c6.button("🚀 Run")

st.markdown('</div>', unsafe_allow_html=True)

if not run_button:
    st.stop()

# ==============================
# LOAD DATA
# ==============================
data = yf.download(symbol, period=period, progress=False)

if data.empty:
    st.error("No data found.")
    st.stop()

# ==============================
# FEATURE ENGINEERING
# ==============================
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()
data["Return"] = data["Close"].pct_change()
data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
data = data.dropna()

features = ["MA20", "MA50"]
X = data[features]
y = data["Target"]

split = int(len(data)*0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, pred)*100, 2)

latest = X.iloc[-1:]
signal = model.predict(latest)
confidence = round(np.max(model.predict_proba(latest))*100,2)

# ==============================
# PERFORMANCE METRICS
# ==============================
data_test = data.iloc[split:].copy()
data_test["Prediction"] = pred
data_test["Strategy_Return"] = data_test["Return"] * data_test["Prediction"]
data_test["Equity"] = (1 + data_test["Strategy_Return"]).cumprod()
data_test["Market"] = (1 + data_test["Return"]).cumprod()

sharpe = round(
    (data_test["Strategy_Return"].mean() /
     data_test["Strategy_Return"].std()) * np.sqrt(252), 2
)

drawdown = round(
    ((data_test["Equity"].cummax() - data_test["Equity"]) /
     data_test["Equity"].cummax()).max()*100,2
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Model Accuracy", f"{accuracy}%")
m2.metric("Sharpe Ratio", sharpe)
m3.metric("Max Drawdown", f"{drawdown}%")
m4.metric("Confidence", f"{confidence}%")

# ==============================
# PRICE CHART
# ==============================
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

fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ==============================
# SIGNAL PANEL
# ==============================
st.subheader("AI Trade Signal")

if signal[0] == 1:
    st.markdown('<div class="signal-buy"><h2>📈 BUY</h2></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="signal-sell"><h2>📉 SELL</h2></div>', unsafe_allow_html=True)

# ==============================
# EQUITY CURVE
# ==============================
if backtest_toggle:
    st.subheader("Strategy vs Market")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data_test.index, y=data_test["Equity"], name="Strategy"))
    fig2.add_trace(go.Scatter(x=data_test.index, y=data_test["Market"], name="Buy & Hold"))
    fig2.update_layout(template="plotly_dark", height=400)

    st.plotly_chart(fig2, use_container_width=True)
