import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Global Trade Dashboard", layout="wide")

st.markdown("""
<style>
body {background-color: #0e1117; color: white;}
</style>
""", unsafe_allow_html=True)

st.title("🌍 AI Global Trade Intelligence Dashboard")

st.warning("⚠️ AI-based analytical tool. Not financial advice.")

# =========================
# GLOBAL MARKET SNAPSHOT
# =========================

st.subheader("📊 Global Market Snapshot")

col1, col2, col3 = st.columns(3)

btc = yf.download("BTC-USD", period="1d", interval="1m")
eth = yf.download("ETH-USD", period="1d", interval="1m")
gold = yf.download("GC=F", period="1d", interval="1m")

col1.metric("Bitcoin (BTC)", f"${round(btc['Close'][-1],2)}")
col2.metric("Ethereum (ETH)", f"${round(eth['Close'][-1],2)}")
col3.metric("Gold", f"${round(gold['Close'][-1],2)}")

st.divider()

# =========================
# AI STOCK ANALYZER
# =========================

st.subheader("🤖 AI Stock Analyzer")

stock_symbol = st.text_input("Enter NSE Stock Symbol (Example: RELIANCE.NS)", "RELIANCE.NS")

if st.button("Analyze Stock"):

    stock = yf.download(stock_symbol, start="2023-01-01")

    stock["MA10"] = stock["Close"].rolling(window=10).mean()
    stock["MA50"] = stock["Close"].rolling(window=50).mean()
    stock["Target"] = (stock["Close"].shift(-1) > stock["Close"]).astype(int)

    stock = stock.dropna()

    X = stock[["MA10", "MA50"]]
    y = stock["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    latest = stock[["MA10", "MA50"]].iloc[-1:]
    prediction = model.predict(latest)
    probability = model.predict_proba(latest)

    confidence = round(np.max(probability) * 100, 2)

    if prediction[0] == 1:
        st.success(f"📈 BUY Signal (Confidence: {confidence}%)")
    else:
        st.error(f"📉 SELL Signal (Confidence: {confidence}%)")

    st.line_chart(stock["Close"])
