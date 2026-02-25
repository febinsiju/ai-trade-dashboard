import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

st.title("🧪 Strategy Backtesting Lab")

symbol = st.text_input("Asset Symbol", "BTC-USD")
period = st.selectbox("Backtest Period", ["1y", "2y", "5y"])

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
