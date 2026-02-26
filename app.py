import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="QuantNova AI Trading Lab",
    layout="wide",
    page_icon="📊"
)

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("QuantNova Navigation")

page = st.sidebar.radio(
    "Go To",
    ["Home", "AI Engine", "Backtesting Lab", "About Us", "Contact", "Follow Us"]
)

# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    st.title("🚀 QuantNova AI Trading Intelligence Platform")

    st.markdown("""
    ### Transforming Market Data into Structured Intelligence

    QuantNova is an AI-powered quantitative research platform designed to analyze
    financial markets using machine learning models and structured validation systems.

    Unlike traditional trading approaches driven by emotion,
    our platform relies on probability-based predictive modeling,
    systematic backtesting, and risk-adjusted performance metrics.
    """)

    st.markdown("---")

    st.header("🔍 What This Platform Does")

    st.markdown("""
    1. Fetches live stock market data.
    2. Engineers technical indicators.
    3. Trains machine learning models.
    4. Predicts next-day market direction.
    5. Validates strategy through historical backtesting.
    6. Measures performance against Buy & Hold.
    """)

    st.markdown("---")

    st.header("🧠 Our Quantitative Methodology")

    st.markdown("""
    The system processes two years of historical data and builds
    supervised learning models using ensemble techniques.

    The AI learns from:
    - Moving averages
    - Price returns
    - Market structure patterns

    The prediction output includes:
    - Directional signal (BUY / SELL)
    - Confidence probability
    - Model accuracy
    """)

    st.markdown("---")

    st.subheader("⚠️ Disclaimer")
    st.info("This platform is developed for educational and research purposes only.")

# ==================================================
# AI ENGINE PAGE
# ==================================================

elif page == "AI Engine":

    st.title("🧠 AI Prediction Engine")

    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT)", "AAPL")

    @st.cache_data
    def load_data(symbol):
        return yf.download(symbol, period="2y")

    data = load_data(stock_symbol)

    if data.empty:
        st.error("Invalid stock symbol.")
        st.stop()

    data["SMA_10"] = data["Close"].rolling(10).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()
    data["Return"] = data["Close"].pct_change()
    data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
    data = data.dropna()

    X = data[["SMA_10", "SMA_50", "Return"]]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=150)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    st.subheader("📊 Model Performance")
    st.metric("Model Accuracy", f"{round(accuracy*100,2)}%")

    latest = X.iloc[-1:].values
    prediction = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0]

    if prediction == 1:
        st.success("📈 AI Signal: BUY")
        st.metric("Confidence", f"{round(prob[1]*100,2)}%")
    else:
        st.error("📉 AI Signal: SELL")
        st.metric("Confidence", f"{round(prob[0]*100,2)}%")

# ==================================================
# BACKTESTING PAGE
# ==================================================

elif page == "Backtesting Lab":

    st.title("📊 Strategy Backtesting Laboratory")

    symbol = st.text_input("Stock Symbol for Backtest", "AAPL")

    data = yf.download(symbol, period="2y")

    data["Return"] = data["Close"].pct_change()
    data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
    data = data.dropna()

    X = data[["Return"]]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    test = data.iloc[-len(X_test):].copy()
    test["Strategy"] = preds * test["Return"]

    test["Cumulative_Market"] = (1 + test["Return"]).cumprod()
    test["Cumulative_Strategy"] = (1 + test["Strategy"]).cumprod()

    fig, ax = plt.subplots()
    ax.plot(test["Cumulative_Market"], label="Buy & Hold")
    ax.plot(test["Cumulative_Strategy"], label="AI Strategy")
    ax.legend()

    st.pyplot(fig)

# ==================================================
# ABOUT US PAGE
# ==================================================

elif page == "About Us":

    st.title("🏢 About QuantNova")

    st.markdown("""
    QuantNova was conceptualized as an academic AI research initiative
    focused on applying machine learning to financial market prediction.

    Our mission is to bridge theoretical data science with
    real-world quantitative trading concepts.

    The platform integrates:
    - Machine learning
    - Financial analytics
    - Risk-adjusted evaluation
    - Automated model retraining concepts
    """)

    st.markdown("### 👨‍💻 Development Team")
    st.write("Lead Developer: Your Name")
    st.write("Research & Modeling: Team Members")
    st.write("Presentation & Analysis: Team Members")

# ==================================================
# CONTACT PAGE
# ==================================================

elif page == "Contact":

    st.title("📞 Contact Us")

    st.markdown("""
    For collaboration, academic research, or project inquiries:
    """)

    st.write("📧 Email: quantnova.ai@gmail.com")
    st.write("📍 Location: Academic Research Lab")
    st.write("🕒 Working Hours: Mon–Fri")

# ==================================================
# FOLLOW US PAGE
# ==================================================

elif page == "Follow Us":

    st.title("🌍 Follow QuantNova")

    st.markdown("""
    Stay connected and follow our research updates.
    """)

    st.write("🔗 LinkedIn: linkedin.com/company/quantnova")
    st.write("🐦 Twitter: twitter.com/quantnova_ai")
    st.write("📸 Instagram: instagram.com/quantnova_ai")

st.markdown("---")
st.markdown(f"© {datetime.datetime.now().year} QuantNova AI Research Lab")
