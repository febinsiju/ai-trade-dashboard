import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="AI Trade Bot X",
    layout="wide",
    page_icon="📈"
)

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
# AUTO REFRESH (NO EXTRA LIB)
# =============================
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = 0

import time
if time.time() - st.session_state.last_refresh > refresh_rate:
    st.session_state.last_refresh = time.time()
    st.rerun()

# =============================
# PROFESSIONAL DARK STYLE
# =============================
st.markdown("""
<style>
body { background-color: #0B0F1C; }
.main { background-color: #0B0F1C; }
.block-container { padding-top: 1rem; }

.buy-box {
    background-color: rgba(0,255,163,0.15);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

.sell-box {
    background-color: rgba(255,60,60,0.15);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

h1, h2, h3 { color: #00FFA3; }
</style>
""", unsafe_allow_html=True)

# =============================
# DATA FUNCTION
# =============================
@st.cache_data(ttl=60)
def get_data(symbol, period="6mo"):
    try:
        return yf.download(
            symbol,
            period=period,
            progress=False,
            auto_adjust=True,
            threads=False
        )
    except:
        return pd.DataFrame()

# =============================
# MAIN LAYOUT
# =============================
st.title("📊 AI Trading Terminal")

left, center, right = st.columns([1.2, 3, 1.2])

# =============================
# LEFT PANEL (Market Stats)
# =============================
with left:
    st.subheader("Market Stats")

    data_5d = get_data(symbol, "5d")

    if not data_5d.empty:
        price = round(float(data_5d["Close"].iloc[-1]), 2)
        prev = float(data_5d["Close"].iloc[-2])
        change = round(((price - prev) / prev) * 100, 2)
        st.metric("Current Price", f"${price}", f"{change}%")
    else:
        st.metric("Current Price", "Unavailable")

# =============================
# CENTER PANEL (Chart)
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
            name="MA20",
            line=dict(width=1)
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["MA50"],
            name="MA50",
            line=dict(width=1)
        ))

        fig.update_layout(
            template="plotly_dark",
            height=650,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("No market data available.")

# =============================
# RIGHT PANEL (AI SIGNAL)
# =============================
with right:
    st.subheader("🤖 AI Signal")

    data_ai = get_data(symbol, "1y")

    if not data_ai.empty:

        data_ai["MA10"] = data_ai["Close"].rolling(10).mean()
        data_ai["MA50"] = data_ai["Close"].rolling(50).mean()
        data_ai["Target"] = (data_ai["Close"].shift(-1) > data_ai["Close"]).astype(int)
        data_ai = data_ai.dropna()

        if len(data_ai) > 50:

            X = data_ai[["MA10", "MA50"]]
            y = data_ai["Target"]

            model = RandomForestClassifier(n_estimators=100)
            model.fit(X, y)

            latest = X.iloc[-1:]
            prediction = model.predict(latest)
            probability = model.predict_proba(latest)
            confidence = round(np.max(probability) * 100, 2)

            if prediction[0] == 1:
                st.markdown('<div class="buy-box"><h2>📈 BUY</h2></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="sell-box"><h2>📉 SELL</h2></div>', unsafe_allow_html=True)

            st.metric("Confidence", f"{confidence}%")

        else:
            st.warning("Not enough AI training data.")

    else:
        st.error("AI data unavailable.")

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
# TRADE HISTORY
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
