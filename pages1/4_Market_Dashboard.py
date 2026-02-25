import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.title("🌍 Market Overview")

symbols = ["^GSPC", "^IXIC", "BTC-USD", "ETH-USD"]

for s in symbols:
    data = yf.download(s, period="5d", progress=False)
    if not data.empty:
        price = round(float(data["Close"].iloc[-1]),2)
        st.metric(s, price)

symbol = st.text_input("Chart Symbol", "BTC-USD")

data = yf.download(symbol, period="6mo", progress=False)

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
