import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide", page_title="AI Trading Platform")

# ===============================
# GLOBAL DARK STYLE
# ===============================
st.markdown("""
<style>
body { background-color: #0E1117; }
.main { background-color: #0E1117; }
.block-container { padding-top: 2rem; }

.hero-title {
    font-size: 48px;
    font-weight: 800;
    color: #00FFA3;
}

.section-box {
    background-color: #141A2A;
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# NAVIGATION MENU
# ===============================
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "AI Prediction", "Backtesting Lab", "Risk Analytics", "Market Overview"]
)

# ===============================
# HOME PAGE
# ===============================
if menu == "Home":

    st.markdown('<div class="hero-title">AI Trading Intelligence Platform</div>', unsafe_allow_html=True)

    st.write("""
    This platform provides institutional-grade AI trading tools.

    We combine:
    • Machine Learning Signal Generation  
    • Historical Backtesting  
    • Risk & Performance Analytics  
    • Real-Time Market Monitoring  

    Use the navigation panel to explore each module.
    """)

    st.divider()

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("AI Prediction Engine")
    st.write("Generate BUY/SELL signals using ML models.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Strategy Backtesting")
    st.write("Test AI performance vs Buy & Hold.")
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# AI PREDICTION PAGE
# ===============================
elif menu == "AI Prediction":

    st.title("AI Prediction Engine")

    symbol = st.text_input("Asset Symbol", "BTC-USD")
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"])

    if st.button("Run Prediction"):

        data = yf.download(symbol, period=period, progress=False)

        if data.empty:
            st.error("No data found.")
            st.stop()

        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
        data = data.dropna()

        X = data[["MA20","MA50"]]
        y = data["Target"]

        model = RandomForestClassifier(n_estimators=300)
        model.fit(X, y)

        latest = X.iloc[-1:]
        signal = model.predict(latest)[0]
        confidence = round(np.max(model.predict_proba(latest))*100,2)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"]
        ))
        fig.update_layout(template="plotly_dark", height=600)

        st.plotly_chart(fig, use_container_width=True)

        if signal == 1:
            st.success(f"BUY Signal (Confidence {confidence}%)")
        else:
            st.error(f"SELL Signal (Confidence {confidence}%)")

# ===============================
# BACKTESTING PAGE
# ===============================
elif menu == "Backtesting Lab":

    st.title("Strategy Backtesting Lab")

    symbol = st.text_input("Asset Symbol", "BTC-USD")
    period = st.selectbox("Backtest Period", ["1y","2y","5y"])

    if st.button("Run Backtest"):

        data = yf.download(symbol, period=period, progress=False)

        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()
        data["Return"] = data["Close"].pct_change()
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
        data = data.dropna()

        split = int(len(data)*0.8)
        train = data[:split]
        test = data[split:]

        model = RandomForestClassifier(n_estimators=300)
        model.fit(train[["MA20","MA50"]], train["Target"])

        test["Prediction"] = model.predict(test[["MA20","MA50"]])
        test["Strategy"] = test["Return"] * test["Prediction"]

        test["Equity"] = (1+test["Strategy"]).cumprod()
        test["Market"] = (1+test["Return"]).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test.index, y=test["Equity"], name="Strategy"))
        fig.add_trace(go.Scatter(x=test.index, y=test["Market"], name="Buy & Hold"))
        fig.update_layout(template="plotly_dark", height=500)

        st.plotly_chart(fig, use_container_width=True)

# ===============================
# RISK ANALYTICS
# ===============================
elif menu == "Risk Analytics":

    st.title("Risk Analytics")

    symbol = st.text_input("Asset Symbol", "BTC-USD")
    period = st.selectbox("Period", ["1y","2y","5y"])

    if st.button("Analyze"):

        data = yf.download(symbol, period=period, progress=False)

        returns = data["Close"].pct_change().dropna()

        sharpe = round((returns.mean()/returns.std())*np.sqrt(252),2)
        volatility = round(returns.std()*np.sqrt(252)*100,2)
        drawdown = round(((data["Close"].cummax()-data["Close"])/data["Close"].cummax()).max()*100,2)

        c1,c2,c3 = st.columns(3)
        c1.metric("Sharpe Ratio", sharpe)
        c2.metric("Annual Volatility", f"{volatility}%")
        c3.metric("Max Drawdown", f"{drawdown}%")

# ===============================
# MARKET OVERVIEW
# ===============================
elif menu == "Market Overview":

    st.title("Market Overview")

    symbols = ["^GSPC","^IXIC","BTC-USD","ETH-USD"]

    cols = st.columns(len(symbols))

    for i,s in enumerate(symbols):
        data = yf.download(s, period="5d", progress=False)
        if not data.empty:
            price = round(float(data["Close"].iloc[-1]),2)
            cols[i].metric(s, price)
