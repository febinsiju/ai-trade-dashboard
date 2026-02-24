import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Global Trade Dashboard", layout="wide")

st.title("🌍 AI Global Trade Intelligence Dashboard")
st.warning("⚠️ AI-based analytical tool. Not financial advice.")

# =========================
# SAFE METRIC FUNCTION
# =========================

def safe_metric(column, label, ticker):
    data = yf.download(ticker, period="5d")
    if not data.empty and "Close" in data.columns:
        price = round(data["Close"].iloc[-1], 2)
        column.metric(label, f"${price}")
    else:
        column.metric(label, "Data unavailable")

# =========================
# GLOBAL MARKET SNAPSHOT
# =========================

st.subheader("📊 Global Market Snapshot")

col1, col2, col3 = st.columns(3)

safe_metric(col1, "Bitcoin (BTC)", "BTC-USD")
safe_metric(col2, "Ethereum (ETH)", "ETH-USD")
safe_metric(col3, "Gold", "GC=F")

st.divider()

# =========================
# AI STOCK ANALYZER
# =========================

st.subheader("🤖 AI Stock Analyzer")

stock_symbol = st.text_input("Enter Stock Symbol (Example: RELIANCE.NS)", "RELIANCE.NS")

if st.button("Analyze Stock"):

    stock = yf.download(stock_symbol, start="2023-01-01")

    if stock.empty or "Close" not in stock.columns:
        st.error("Unable to fetch stock data. Please check symbol.")
    else:
        stock["MA10"] = stock["Close"].rolling(window=10).mean()
        stock["MA50"] = stock["Close"].rolling(window=50).mean()
        stock["Target"] = (stock["Close"].shift(-1) > stock["Close"]).astype(int)

        stock = stock.dropna()

        if len(stock) < 50:
            st.error("Not enough data to analyze.")
        else:
            X = stock[["MA10", "MA50"]]
            y = stock["Target"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

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
