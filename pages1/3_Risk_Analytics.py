import streamlit as st
import yfinance as yf
import numpy as np

st.title("📊 Risk Analytics Dashboard")

symbol = st.text_input("Asset Symbol", "BTC-USD")
period = st.selectbox("Period", ["1y", "2y", "5y"])

if st.button("Analyze Risk"):

    data = yf.download(symbol, period=period, progress=False)

    returns = data["Close"].pct_change().dropna()

    sharpe = round((returns.mean()/returns.std()) * np.sqrt(252), 2)
    volatility = round(returns.std() * np.sqrt(252) * 100, 2)
    drawdown = round(((data["Close"].cummax() - data["Close"]) /
                      data["Close"].cummax()).max()*100,2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpe Ratio", sharpe)
    col2.metric("Annual Volatility", f"{volatility}%")
    col3.metric("Max Drawdown", f"{drawdown}%")
