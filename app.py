import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="AI Trade Bot X",
    layout="wide",
    page_icon="📈"
)

# =============================
# AUTO REFRESH (5 sec)
# =============================
st.experimental_set_query_params(refresh=str(time.time()))

# =============================
# PROFESSIONAL EXCHANGE CSS
# =============================
st.markdown("""
<style>
body { background-color: #0B0F1C; }
.main { background-color: #0B0F1C; }

.block-container {
    padding-top: 1rem;
}

.card {
    background-color: #141A2A;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 0px 30px rgba(0,255,170,0.08);
}

.buy-box {
    background-color: rgba(0, 255, 170, 0.15);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

.sell-box {
    background-color: rgba(255, 60, 60, 0.15);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

h1, h2, h3 {
    color: #00FFA3;
}

.stButton>button {
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =============================
# DATA FUNCTION
# =============================
def get_data(symbol, period="3mo"):
    try:
        data = yf.download(
            symbol,
            period=period,
            progress=False,
            auto_adjust=True,
            threads=False
        )
        return data
    except:
        return pd.DataFrame()

# =============================
# SIDEBAR
# =============================
st.sidebar.title("⚡ AI TRADE BOT X")

symbol = st.sidebar.selectbox(
    "Select Asset",
    ["BTC-USD", "ETH-USD", "SOL-USD", "AAPL", "TSLA"]
)

refresh_rate = st.sidebar.slider("Auto Refresh (seconds)", 5, 60, 10)

# =============================
# MAIN DASHBOARD LAYOUT
# =============================
st.title("📊 AI Trading Terminal")

left, center, right = st.columns([1.2, 3, 1.2])

# =============================
# MARKET STATS (LEFT)
# =============================
with left:
    st.markdown("### Market Stats")
    data = get_data(symbol, "5d")

    if not data.empty:
        price = round(float(data["Close"].iloc[-1]), 2)
        prev = float(data["Close"].iloc[-2])
        change = round(((price - prev) / prev) * 100, 2)

        st.metric("Current Price", f"${price}", f"{change}%")
    else:
        st.metric("Current Price", "Unavailable")

# =============================
# MAIN CHART
# =============================
with center:

    data = get_data(symbol, "6mo")

    if not data.empty:

        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price"
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["MA20"],
            line=dict(width=1),
            name="MA20"
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["MA50"],
            line=dict(width=1),
            name="MA50"
        ))

        fig.update_layout(
            template="plotly_dark",
            height=650,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("No data available.")

# =============================
# AI SIGNAL PANEL (RIGHT)
# =============================
with right:

    st.markdown("### 🤖 AI Signal")

    data = get_data(symbol, "1y")

    if not data.empty:

        data["MA10"] = data["Close"].rolling(10).mean()
        data["MA50"] = data["Close"].rolling(50).mean()
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
        data = data.dropna()

        X = data[["MA10", "MA50"]]
        y = data["Target"]

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)

        latest = data[["MA10", "MA50"]].iloc[-1:]
        prediction = model.predict(latest)
        probability = model.predict_proba(latest)
        confidence = round(np.max(probability) * 100, 2)

        if prediction[0] == 1:
            st.markdown('<div class="buy-box"><h2>📈 BUY</h2></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sell-box"><h2>📉 SELL</h2></div>', unsafe_allow_html=True)

        st.metric("Confidence", f"{confidence}%")

    else:
        st.error("AI unavailable")

# =============================
# ORDER PANEL
# =============================
st.divider()
st.subheader("💹 Execute Trade")

col1, col2, col3 = st.columns(3)

amount = col1.number_input("Amount", min_value=0.0, value=1.0)
col2.button("🟢 BUY", use_container_width=True)
col3.button("🔴 SELL", use_container_width=True)

# =============================
# TRADE HISTORY TABLE
# =============================
st.divider()
st.subheader("📜 Trade History")

history = pd.DataFrame({
    "Asset": [symbol]*5,
    "Type": ["BUY", "SELL", "BUY", "BUY", "SELL"],
    "Price": np.random.uniform(100, 50000, 5).round(2),
    "PnL %": np.random.uniform(-5, 8, 5).round(2)
})

st.dataframe(history, use_container_width=True)

# =============================
# AUTO REFRESH LOOP
# =============================
time.sleep(refresh_rate)
st.rerun()
