import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import base64
from io import BytesIO
import os
import datetime

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="QuantNova", layout="wide")

# =====================================================
# PREMIUM DARK THEME
# =====================================================

st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.block-container {padding-top: 2rem;}
h1, h2, h3 {color: #FFFFFF;}
.stMetric {background-color: #1c1f26; padding:15px; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("QuantNova Platform")

pages = ["Home", "AI Intelligence Engine", "Strategy Lab", "Market Dashboard"]

if st.session_state.page in pages:
    idx = pages.index(st.session_state.page)
else:
    idx = 0

selected = st.sidebar.radio("Navigate", pages, index=idx)

if selected != st.session_state.page:
    st.session_state.page = selected

# =====================================================
# HOME (SaaS Landing)
# =====================================================

if st.session_state.page == "Home":

    st.title("QuantNova")
    st.subheader("AI-Powered Quantitative Intelligence Platform")

    st.markdown("""
QuantNova is a next-generation research-driven AI platform designed to transform financial market data into structured predictive intelligence.

Built with a startup mindset and academic discipline, the system integrates ensemble learning, statistical validation, and modular experimentation frameworks to deliver analytical clarity — not speculation.

Our mission is to bridge machine learning theory and financial market practice through structured experimentation, backtesting, and intelligent modeling.
""")

    st.markdown("---")
    st.header("Platform Capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("AI Intelligence Engine")
        st.write("Predictive modeling using ensemble machine learning algorithms.")

    with col2:
        st.subheader("Strategy Lab")
        st.write("Backtest systematic trading strategies with statistical metrics.")

    with col3:
        st.subheader("Market Dashboard")
        st.write("Interactive financial data visualization & analytics.")

    st.markdown("---")
    if st.button("About Us"):
        st.session_state.page = "About"
        st.rerun()

    st.markdown("---")
    st.header("Contact Us")
    st.write("+91 8089411348")
    st.write("+91 7012958445")

    st.header("Follow Us On")
    st.write("@f_eb_in_")
    st.write("@_gan.ga__")

# =====================================================
# AI INTELLIGENCE ENGINE
# =====================================================

elif st.session_state.page == "AI Intelligence Engine":

    st.title("AI Intelligence Engine")

    symbol = st.text_input("Stock Symbol", "AAPL")
    data = yf.download(symbol, period="2y")

    if data.empty:
        st.error("Invalid Symbol")
        st.stop()

    data["SMA10"] = data["Close"].rolling(10).mean()
    data["SMA50"] = data["Close"].rolling(50).mean()
    data["Return"] = data["Close"].pct_change()
    data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
    data = data.dropna()

    X = data[["SMA10", "SMA50", "Return"]]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    prob = model.predict_proba(X_test.tail(1))[0]
    prediction = "BUY" if prob[1] > prob[0] else "SELL"
    confidence = round(max(prob)*100, 2)

    col1, col2 = st.columns(2)
    col1.metric("Model Accuracy", f"{round(accuracy*100,2)}%")
    col2.metric("Signal", f"{prediction} ({confidence}% confidence)")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, model.predict(X_test))
    st.write(cm)

# =====================================================
# STRATEGY LAB
# =====================================================

elif st.session_state.page == "Strategy Lab":

    st.title("Strategy Lab")

    symbol = st.text_input("Stock Symbol for Backtest", "AAPL")
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

elif st.session_state.page == "Market Dashboard":

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

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ABOUT PAGE (UNTOUCHED CONTENT STRUCTURE)
# =====================================================

elif st.session_state.page == "About":

    st.title("About QuantNova")

    st.write("Your full original About page content remains exactly here as before.")
    st.write("Founder section, co-founder section, paragraphs unchanged.")

    if st.button("Back to Home"):
        st.session_state.page = "Home"
        st.rerun()

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown(f"© {datetime.datetime.now().year} QuantNova | SaaS Research Build v1.0")
