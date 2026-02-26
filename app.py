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
# CONFIG
# ==================================================

st.set_page_config(
    page_title="QuantNova AI Trading Intelligence",
    layout="wide",
    page_icon="📊"
)

# ==================================================
# SESSION STATE FOR PAGE SWITCHING
# ==================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

def switch_page(page_name):
    st.session_state.page = page_name

# ==================================================
# SIDEBAR NAV
# ==================================================

st.sidebar.title("QuantNova Navigation")

sidebar_choice = st.sidebar.radio(
    "Navigate",
    ["Home", "AI Engine", "Backtesting Lab"]
)

st.session_state.page = sidebar_choice

# ==================================================
# HOME PAGE (LONG PROFESSIONAL LANDING)
# ==================================================

if st.session_state.page == "Home":

    st.title("🚀 QuantNova AI Trading Intelligence Platform")

    st.markdown("""
    ## Redefining Financial Decision Intelligence

    QuantNova is a next-generation quantitative research platform engineered
    to transform raw financial data into structured, machine-driven insight.

    Modern markets operate at extraordinary speed and complexity.
    Traditional human-driven analysis struggles to process the scale
    and depth of available data.

    QuantNova leverages machine learning, structured validation,
    and systematic modeling to generate probabilistic trading signals.
    """)

    st.markdown("---")

    st.header("🌍 Why Quantitative Intelligence Matters")

    st.markdown("""
    Financial markets are dynamic, multi-factor systems influenced by:

    - Macroeconomic indicators  
    - Institutional capital flows  
    - Algorithmic execution systems  
    - Behavioral biases  

    QuantNova removes emotional bias and replaces it with
    systematic probability-based reasoning.
    """)

    st.markdown("---")

    st.header("🧠 AI Architecture Overview")

    st.markdown("""
    Our predictive framework follows a structured pipeline:

    ### 1️⃣ Data Acquisition
    - Live stock data retrieval
    - Historical market structures

    ### 2️⃣ Feature Engineering
    - Moving averages
    - Return structures
    - Trend behavior metrics

    ### 3️⃣ Model Training
    - Supervised learning
    - Ensemble classification models
    - Pattern recognition logic

    ### 4️⃣ Validation
    - Historical backtesting
    - Accuracy scoring
    - Risk-adjusted evaluation

    ### 5️⃣ Decision Output
    - BUY / SELL signal
    - Confidence probability
    - Performance benchmarking
    """)

    st.markdown("---")

    st.header("📊 Risk-Adjusted Philosophy")

    st.markdown("""
    Raw returns alone do not define performance.

    QuantNova compares AI strategy output against
    baseline Buy & Hold structures and evaluates:

    - Structural stability
    - Drawdown sensitivity
    - Consistency of signal generation
    """)

    st.markdown("---")

    st.header("🔮 Future Vision")

    st.markdown("""
    Our roadmap includes:

    - Automated daily retraining
    - Multi-asset support (stocks + crypto)
    - Deep learning integration
    - Reinforcement learning agents
    - Institutional-level performance metrics
    """)

    st.markdown("---")

    st.subheader("⚠️ Disclaimer")
    st.info("This platform is developed for academic and research demonstration purposes only.")

    # ================= FOOTER ==================

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

    symbol = st.text_input("Enter Stock Symbol", "AAPL")

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
        st.metric("Confidence", f"{round(prob[1]*100,2)}%")
    else:
        st.error("📉 SELL Signal")
        st.metric("Confidence", f"{round(prob[0]*100,2)}%")

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
    QuantNova was built as an academic AI research initiative
    focused on merging financial analytics with machine learning.

    Our objective is to demonstrate how structured AI systems
    can assist in disciplined decision-making.

    Developed by:
    - Project Lead: Your Name
    - AI Research Team: Team Members
    - Presentation & Strategy: Team Members
    """)

# ==================================================
# CONTACT
# ==================================================

elif st.session_state.page == "Contact":

    st.title("📞 Contact Us")

    st.write("📧 Email: quantnova.ai@gmail.com")
    st.write("📍 Location: Academic Research Initiative")
    st.write("🕒 Availability: Mon–Fri")

# ==================================================
# FOLLOW US
# ==================================================

elif st.session_state.page == "Follow Us":

    st.title("🌍 Follow QuantNova")

    st.write("🔗 LinkedIn: linkedin.com/company/quantnova")
    st.write("🐦 Twitter: twitter.com/quantnova_ai")
    st.write("📸 Instagram: instagram.com/quantnova_ai")

# ==================================================
# FOOTER COPYRIGHT
# ==================================================

st.markdown("---")
st.markdown(f"© {datetime.datetime.now().year} QuantNova AI Research Lab")
