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
    page_title="QuantNova AI Trading Intelligence",
    layout="wide",
    page_icon="📊"
)

# ==================================================
# GLOBAL STYLING (BIG PROFESSIONAL TYPOGRAPHY)
# ==================================================

st.markdown("""
<style>

.big-title {
    font-size: 52px;
    font-weight: 800;
    margin-bottom: 25px;
}

.section-heading {
    font-size: 38px;
    font-weight: 700;
    margin-top: 70px;
    margin-bottom: 30px;
}

.big-text {
    font-size: 20px;
    line-height: 1.9;
    margin-bottom: 25px;
}

.big-points {
    font-size: 20px;
    margin-bottom: 15px;
}

.footer-heading {
    font-size: 24px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# SESSION STATE
# ==================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

def switch_page(page_name):
    st.session_state.page = page_name

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("QuantNova Navigation")

sidebar_choice = st.sidebar.radio(
    "Navigate",
    ["Home", "AI Engine", "Backtesting Lab"]
)

st.session_state.page = sidebar_choice

# ==================================================
# HOME PAGE
# ==================================================

if st.session_state.page == "Home":

    st.markdown('<div class="big-title">🚀 QuantNova AI Trading Intelligence Platform</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="big-text">
    QuantNova is a next-generation quantitative research platform engineered
    to convert raw financial market data into structured, machine-driven intelligence.

    In today’s algorithm-dominated markets, traditional emotional decision-making
    is replaced by structured, probability-based reasoning.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-heading">🌍 Why Quantitative Intelligence Matters</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="big-points">• Eliminates emotional bias in trading decisions</div>
    <div class="big-points">• Processes large-scale historical datasets</div>
    <div class="big-points">• Detects structural price behavior patterns</div>
    <div class="big-points">• Applies supervised machine learning models</div>
    <div class="big-points">• Compares AI results vs traditional strategies</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-heading">🧠 AI System Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="big-points">1️⃣ Data Acquisition – Live stock data retrieval</div>
    <div class="big-points">2️⃣ Feature Engineering – Moving averages & return modeling</div>
    <div class="big-points">3️⃣ Model Training – Random Forest ensemble learning</div>
    <div class="big-points">4️⃣ Validation – Structured backtesting workflows</div>
    <div class="big-points">5️⃣ Decision Output – BUY/SELL signal with confidence probability</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-heading">📊 Risk-Adjusted Philosophy</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="big-text">
    True performance is not measured by raw return alone.

    QuantNova benchmarks AI strategy output against Buy & Hold
    while evaluating structural stability, risk exposure,
    and consistency of signal generation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-heading">🔮 Future Development Roadmap</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="big-points">• Automated daily model retraining</div>
    <div class="big-points">• Multi-asset support (Stocks & Crypto)</div>
    <div class="big-points">• Deep learning integration</div>
    <div class="big-points">• Reinforcement learning agents</div>
    <div class="big-points">• Institutional-grade performance metrics</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.info("⚠️ Developed for academic research and demonstration purposes only.")

    # FOOTER
    st.markdown("---")
    st.markdown("### 📌 Explore More")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🏢 About Us"):
            switch_page("About Us")

    with col2:
        if st.button("📞 Contact"):
            switch_page("Contact")

    with col3:
        if st.button("🌍 Follow Us"):
            switch_page("Follow Us")

# ==================================================
# AI ENGINE
# ==================================================

elif st.session_state.page == "AI Engine":

    st.title("🧠 AI Prediction Engine")

    symbol = st.text_input("Enter Stock Symbol (Example: AAPL)", "AAPL")

    data = yf.download(symbol, period="2y")

    if data.empty:
        st.error("Invalid Stock Symbol")
        st.stop()

    data["SMA10"] = data["Close"].rolling(10).mean()
    data["SMA50"] = data["Close"].rolling(50).mean()
    data["Return"] = data["Close"].pct_change()
    data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
    data = data.dropna()

    X = data[["SMA10", "SMA50", "Return"]]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=150)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    st.metric("Model Accuracy", f"{round(accuracy*100,2)}%")

    latest = X.iloc[-1:].values
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0]

    if pred == 1:
        st.success("📈 BUY Signal")
        st.metric("Confidence Level", f"{round(prob[1]*100,2)}%")
    else:
        st.error("📉 SELL Signal")
        st.metric("Confidence Level", f"{round(prob[0]*100,2)}%")

# ==================================================
# BACKTESTING
# ==================================================

elif st.session_state.page == "Backtesting Lab":

    st.title("📊 Strategy Backtesting Laboratory")

    symbol = st.text_input("Stock Symbol", "AAPL")

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

    test["Market"] = (1 + test["Return"]).cumprod()
    test["AI"] = (1 + test["Strategy"]).cumprod()

    fig, ax = plt.subplots()
    ax.plot(test["Market"], label="Buy & Hold")
    ax.plot(test["AI"], label="AI Strategy")
    ax.legend()
    st.pyplot(fig)

# ==================================================
# ABOUT US
# ==================================================

elif st.session_state.page == "About Us":

    st.title("🏢 About QuantNova")

    st.markdown("""
    QuantNova was conceptualized as an academic AI research initiative
    designed to demonstrate the power of machine learning in financial markets.

    Our mission is to merge data science, financial modeling,
    and structured decision intelligence into a unified research platform.
    """)

# ==================================================
# CONTACT
# ==================================================

elif st.session_state.page == "Contact":

    st.title("📞 Contact Us")
    st.write("📧 Email: quantnova.ai@gmail.com")
    st.write("📍 Location: Academic Research Initiative")

# ==================================================
# FOLLOW US
# ==================================================

elif st.session_state.page == "Follow Us":

    st.title("🌍 Follow QuantNova")
    st.write("🔗 LinkedIn: linkedin.com/company/quantnova")
    st.write("🐦 Twitter: twitter.com/quantnova_ai")
    st.write("📸 Instagram: instagram.com/quantnova_ai")

# ==================================================
# COPYRIGHT
# ==================================================

st.markdown("---")
st.markdown(f"© {datetime.datetime.now().year} QuantNova AI Research Lab")
