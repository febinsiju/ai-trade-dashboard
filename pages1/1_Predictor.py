import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

st.title("📈 AI Prediction Engine")

symbol = st.text_input("Asset Symbol", "BTC-USD")
period = st.selectbox("Historical Period", ["6mo", "1y", "2y", "5y"])

if st.button("Run Prediction"):

    data = yf.download(symbol, period=period, progress=False)

    if data.empty:
        st.error("No data found.")
        st.stop()

    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data = data.dropna()

    X = data[["MA20", "MA50"]]
    y = data["Target"]

    model = RandomForestClassifier(n_estimators=300)
    model.fit(X, y)

    latest = X.iloc[-1:]
    signal = model.predict(latest)[0]
    confidence = round(np.max(model.predict_proba(latest)) * 100, 2)

    col1, col2 = st.columns([3,1])

    with col1:
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

    with col2:
        st.subheader("AI Signal")
        if signal == 1:
            st.success("📈 BUY")
        else:
            st.error("📉 SELL")
        st.metric("Confidence", f"{confidence}%")
