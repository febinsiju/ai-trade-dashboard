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
The AI Trading Intelligence Platform is a quantitative research environment 
designed to transform raw market data into structured, data-driven trading insights.

Modern financial markets generate enormous amounts of information every second. 
Price fluctuations, volatility shifts, liquidity changes, and trend formations 
all create complex patterns that are difficult to interpret through manual observation alone.

This platform applies machine learning techniques and statistical modeling 
to systematically analyze those patterns and extract actionable signals.
""")

    st.write("""
Unlike traditional trading dashboards that only visualize price movements, 
this system integrates predictive analytics, performance validation, and 
risk measurement into one unified framework.

The objective is not to guarantee profit — no system can do that — 
but to reduce emotional bias and provide probabilistic insight 
into potential market direction.
""")

    st.divider()

    st.header("Our Quantitative Framework")

    st.write("""
The platform is built upon a structured research methodology similar to 
those used within institutional quantitative trading teams.

It consists of four interconnected layers:
""")

    st.write("""
**1. Data Acquisition Layer**  
Market data is retrieved from reliable financial sources and structured 
for analytical processing. Historical price series are cleaned, transformed, 
and prepared for feature extraction.

**2. Feature Engineering Layer**  
Technical indicators such as moving averages, momentum metrics, 
and return-based statistics are calculated to represent market behavior 
in a format suitable for machine learning models.

**3. Machine Learning Layer**  
Classification algorithms analyze historical relationships between features 
and future price movements. The model produces probabilistic BUY or SELL signals 
based on learned market structure.

**4. Validation & Risk Assessment Layer**  
Performance is evaluated through historical backtesting. Risk-adjusted metrics 
such as Sharpe Ratio and Maximum Drawdown are calculated to assess sustainability.
""")

    st.divider()

    st.header("Platform Capabilities")

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("AI Prediction Engine")
    st.write("""
The AI Prediction Engine applies supervised machine learning models to 
detect short-term directional bias in financial assets.

Users can input stocks, cryptocurrencies, or indices and generate 
a structured BUY or SELL signal with an associated confidence level.

The purpose of this engine is exploratory — enabling users to 
analyze how algorithmic models interpret market data under different conditions.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Strategy Backtesting Laboratory")
    st.write("""
Backtesting is a critical component of quantitative research. 
Before trusting any signal, it must be evaluated against historical market conditions.

This module compares AI-driven strategy performance against 
a traditional Buy & Hold approach, allowing users to measure:

• Equity curve growth  
• Relative performance  
• Compounded returns  
• Structural consistency  

This process ensures that strategies are evaluated systematically rather than emotionally.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Risk & Performance Analytics")
    st.write("""
Returns alone do not define a successful strategy. 
Risk exposure plays an equally important role.

The Risk Analytics module calculates institutional-grade metrics including:

• Sharpe Ratio (risk-adjusted return)  
• Annualized Volatility  
• Maximum Drawdown  
• Return Stability Indicators  

These metrics provide insight into whether performance is sustainable 
under real-world market fluctuations.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Market Intelligence Dashboard")
    st.write("""
The Market Overview module provides a consolidated view of 
major financial indices and digital assets.

Integrated candlestick visualization allows users to observe:

• Recent price structure  
• Short-term momentum  
• Market direction trends  

This dashboard serves as a situational awareness tool 
for monitoring broader financial conditions while conducting research.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    st.header("Intended Use & Philosophy")

    st.write("""
This platform is designed for educational, research, and exploratory purposes.

Financial markets are inherently uncertain, and no predictive model 
can eliminate risk entirely. Instead, the objective of this system is to:

• Reduce emotional decision-making  
• Introduce structured analytical thinking  
• Provide probabilistic insight  
• Encourage disciplined strategy evaluation  

Quantitative trading is not about certainty — it is about managing probabilities 
within a controlled risk framework.
""")

    st.write("""
By combining machine learning, technical analysis, and structured validation, 
the AI Trading Intelligence Platform offers a foundational environment 
for systematic trading research.

Use the navigation menu to explore each module in depth 
and begin your analytical workflow.
""")

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
