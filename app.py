import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="AI Trade Predictor",
    layout="wide",
    page_icon="📈"
)

# =============================
# PROFESSIONAL DARK STYLE
# =============================
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

# =============================
# HEADER
# =============================
st.title("📊 AI Trading Prediction Terminal")

st.markdown("Analyze any stock or crypto asset using AI-driven prediction.")

# =============================
# USER INPUT SECTION
# =============================
col1, col2, col3 = st.columns([2,1,1])

symbol = col1.text_input("Enter Asset Symbol", "BTC-USD")
period = col2.selectbox("Historical Period", ["6mo", "1y", "2y", "5y"])
analyze = col3.button("🚀 Predict")

# =============================
# MAIN LOGIC
# =============================
if analyze:

    with st.spinner("Fetching market data and running AI model..."):

        data = yf.download(
            symbol,
            period=period,
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if data.empty:
            st.error("No data found for this symbol.")
            st.stop()

        # Feature Engineering
        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()
        data["RSI"] = 100 - (100 / (1 + data["Close"].pct_change().rolling(14).mean()))
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
        data = data.dropna()

        if len(data) < 100:
            st.error("Not enough historical data.")
            st.stop()

        # Train Model
        X = data[["MA20", "MA50"]]
        y = data["Target"]

        split = int(len(data) * 0.8)

        X_train = X[:split]
        y_train = y[:split]

        model = RandomForestClassifier(n_estimators=200)
        model.fit(X_train, y_train)

        latest = X.iloc[-1:]
        prediction = model.predict(latest)
        probability = model.predict_proba(latest)
        confidence = round(np.max(probability) * 100, 2)

        # =============================
        # LAYOUT
        # =============================
        left, right = st.columns([3,1])

        # Chart
        with left:
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
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=30, b=10)
            )

            st.plotly_chart(fig, use_container_width=True)

        # Signal Panel
        with right:
            st.subheader("AI Prediction")

            if prediction[0] == 1:
                st.markdown('<div class="signal-buy"><h2>📈 BUY</h2></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="signal-sell"><h2>📉 SELL</h2></div>', unsafe_allow_html=True)

            st.metric("Confidence Level", f"{confidence}%")

            current_price = round(float(data["Close"].iloc[-1]), 2)
            st.metric("Current Price", f"${current_price}")
