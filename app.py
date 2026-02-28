import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import ta
import datetime

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="QuantNova", layout="wide")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("QuantNova Platform")

pages = [
    "Home",
    "AI Intelligence Engine",
    "Real-Time Signal Engine",
    "Strategy Lab",
    "Market Dashboard",
    "About"
]

page = st.sidebar.radio("Navigate", pages)

# =====================================================
# HOME (UNCHANGED INTRO)
# =====================================================

if page == "Home":

    st.title("QuantNova")
    st.subheader("AI-Powered Quantitative Intelligence Platform")

    st.markdown("""The Operating System for AI-Driven Market Intelligence

QuantNova is not entering the financial analytics industry.
It is architecting the layer that will sit beneath it.

We are building an AI-native quantitative intelligence operating system designed to convert global market complexity into structured, self-evolving, probabilistic decision architecture.

Financial markets generate terabytes of data every second — but raw data is noise without structured intelligence. The next era of dominance will not belong to those who see more data. It will belong to those who can model uncertainty, quantify asymmetry, validate structure, and adapt faster than systemic change.

QuantNova is engineered for that era.

At its foundation lies a multi-layer intelligence stack:

• Adaptive ensemble learning systems
• Probabilistic modeling and uncertainty quantification
• Statistical validation and structural integrity engines
• Modular experimentation frameworks
• Scalable backtesting and performance analytics cores
• Infrastructure-ready deployment architecture

Every signal is measurable.
Every prediction is probabilistic.
Every model is reproducible.
Every system is expandable.

QuantNova is not a dashboard.
It is not a signal bot.
It is not a retail trading assistant.

It is a scalable intelligence infrastructure capable of evolving into:

• Cross-asset AI prediction networks
• Institutional-grade strategy simulation ecosystems
• Autonomous model evolution engines
• High-frequency data structuring pipelines
• Enterprise API intelligence layers
• AI-powered hedge fund architecture

Markets are increasingly algorithmic. Capital is increasingly automated. Decision cycles are increasingly compressed.

The companies that win in this environment will not build tools — they will build infrastructure.

QuantNova is being designed as that infrastructure.

Not to compete for attention.
But to become foundational.

Where others optimize indicators, we engineer intelligence systems.
Where others search for signals, we construct predictive architecture.
Where others iterate features, we architect dominance.

QuantNova is not a startup experimenting with finance.
It is a research-driven AI systems company entering financial intelligence as its first domain of deployment.""")

# =====================================================
# AI INTELLIGENCE ENGINE
# =====================================================

elif page == "AI Intelligence Engine":

    st.title("AI Intelligence Engine")

    symbol = st.text_input("Stock Symbol", "AAPL")
    data = yf.download(symbol, period="2y", interval="1d")

    if data.empty:
        st.error("Invalid symbol")
        st.stop()

    # Feature Engineering
    data["SMA20"] = data["Close"].rolling(20).mean()
    data["SMA50"] = data["Close"].rolling(50).mean()
    data["RSI"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
    data["ATR"] = ta.volatility.AverageTrueRange(
        data["High"], data["Low"], data["Close"]
    ).average_true_range()
    data["MACD"] = ta.trend.MACD(data["Close"]).macd()
    data["Volatility"] = data["Close"].pct_change().rolling(10).std()

    data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
    data.dropna(inplace=True)

    features = ["SMA20", "SMA50", "RSI", "ATR", "MACD", "Volatility"]

    X = data[features]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    model = RandomForestClassifier(n_estimators=300)
    calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv=3)

    calibrated_model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, calibrated_model.predict(X_test))

    last_prob = calibrated_model.predict_proba(X_test.tail(1))[0]

    prediction = "BUY" if last_prob[1] > last_prob[0] else "SELL"
    confidence = round(max(last_prob) * 100, 2)

    # Risk Metrics
    returns = data["Close"].pct_change()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    cumulative = (1 + returns).cumprod()
    drawdown = (cumulative / cumulative.cummax() - 1).min()

    expected_move = round(data["ATR"].iloc[-1], 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", f"{round(accuracy*100,2)}%")
    col2.metric("Signal", prediction)
    col3.metric("Confidence", f"{confidence}%")

    col4, col5, col6 = st.columns(3)
    col4.metric("Sharpe Ratio", round(sharpe,2))
    col5.metric("Max Drawdown", f"{round(drawdown*100,2)}%")
    col6.metric("Expected Move (ATR)", expected_move)

# =====================================================
# REAL-TIME SIGNAL ENGINE
# =====================================================

elif page == "Real-Time Signal Engine":

    st.title("Real-Time Signal Engine")

    symbol = st.text_input("Stock Symbol", "AAPL")

    data = yf.download(symbol, period="5d", interval="5m")

    if data.empty:
        st.error("Invalid symbol")
        st.stop()

    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

    momentum = data["Close"].pct_change().rolling(5).mean().iloc[-1]
    signal = "BUY" if momentum > 0 else "SELL"

    st.metric("Live Momentum Signal", signal)

# =====================================================
# STRATEGY LAB
# =====================================================

elif page == "Strategy Lab":

    st.title("Strategy Lab")

    symbol = st.text_input("Stock Symbol", "AAPL")
    data = yf.download(symbol, period="2y")

    short = st.slider("Short SMA", 5, 30, 10)
    long = st.slider("Long SMA", 20, 100, 50)

    data["Short"] = data["Close"].rolling(short).mean()
    data["Long"] = data["Close"].rolling(long).mean()

    data["Signal"] = np.where(data["Short"] > data["Long"], 1, 0)
    data["Returns"] = data["Close"].pct_change()
    data["Strategy"] = data["Signal"].shift(1) * data["Returns"]

    cumulative = (1 + data["Strategy"]).cumprod()

    st.line_chart(cumulative)

    total_return = round((cumulative.iloc[-1] - 1) * 100, 2)
    st.metric("Total Strategy Return", f"{total_return}%")

# =====================================================
# MARKET DASHBOARD
# =====================================================

elif page == "Market Dashboard":

    st.title("Market Dashboard")

    symbol = st.text_input("Stock Symbol", "AAPL")
    data = yf.download(symbol, period="6mo")

    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ABOUT (UNCHANGED)
# =====================================================

elif page == "About":

    st.title("About QuantNova")

    st.markdown("""
    QuantNova was conceived as an academic research initiative by students of
    Computer Science and Engineering (CSE B S2) at TocH Institute Of Science And Technology (TIST),
    Ernakulam, Kerala.

    The project reflects an ambition to bridge theoretical machine learning concepts
    with practical financial data modeling. It represents a structured effort to build,
    test, validate, and continuously refine predictive intelligence systems within a
    disciplined research framework.
    """)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown(f"© {datetime.datetime.now().year} QuantNova | Research Build v2.0")
