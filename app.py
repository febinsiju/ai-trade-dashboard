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
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="QuantNova AI Trading Intelligence",
    layout="wide"
)

# ==================================================
# PROFESSIONAL TYPOGRAPHY STYLING
# ==================================================

st.markdown("""
<style>

.big-title {
    font-size: 54px;
    font-weight: 800;
    margin-bottom: 30px;
}

.section-heading {
    font-size: 40px;
    font-weight: 700;
    margin-top: 80px;
    margin-bottom: 30px;
}

.large-paragraph {
    font-size: 21px;
    line-height: 2;
    margin-bottom: 35px;
    text-align: justify;
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

st.sidebar.title("Navigation")

sidebar_choice = st.sidebar.radio(
    "Select Section",
    ["Home", "AI Engine", "Backtesting Laboratory"]
)

st.session_state.page = sidebar_choice

# ==================================================
# HOME PAGE
# ==================================================

if st.session_state.page == "Home":

    st.markdown('<div class="big-title">QuantNova AI Trading Intelligence Platform</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="large-paragraph">
    QuantNova is a quantitative research platform designed to transform raw financial
    market data into structured, machine-driven intelligence. Modern financial markets
    operate with extraordinary speed and complexity, where algorithmic execution,
    institutional capital flows, and large-scale data processing dominate price behavior.
    Traditional discretionary analysis struggles to keep pace with this evolving structure.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="large-paragraph">
    The platform applies supervised machine learning models to historical price
    structures, extracting meaningful patterns and probabilistic relationships.
    Instead of relying on emotion or speculation, QuantNova emphasizes statistical
    reasoning, structured validation, and disciplined analytical frameworks.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Quantitative Intelligence Framework</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="large-paragraph">
    The system operates through a structured analytical pipeline beginning with
    live financial data acquisition. Historical market data is transformed into
    engineered features such as moving averages and return dynamics. These features
    are processed through ensemble learning algorithms that identify recurring
    structural relationships within price movements.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="large-paragraph">
    Once trained, the model produces directional outputs indicating potential
    upward or downward movement in the subsequent trading session. Each prediction
    is accompanied by a probability measure reflecting the model’s internal
    confidence. This allows interpretation within a probabilistic framework
    rather than deterministic certainty.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Risk-Adjusted Validation Philosophy</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="large-paragraph">
    Performance is not evaluated through raw return alone. QuantNova benchmarks
    its AI-generated strategy against traditional Buy and Hold methodologies
    using structured backtesting procedures. By examining cumulative performance,
    signal consistency, and comparative structural growth, the system demonstrates
    how algorithmic reasoning may outperform passive allocation under certain
    market conditions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="large-paragraph">
    This validation framework ensures that every predictive output is supported
    by historical evidence, thereby reinforcing disciplined analytical reasoning.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("This platform is developed for academic research and demonstration purposes.")

    # FOOTER
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("About Us"):
            switch_page("About Us")

    with col2:
        if st.button("Contact"):
            switch_page("Contact")

    with col3:
        if st.button("Follow Us"):
            switch_page("Follow Us")

# ==================================================
# AI ENGINE
# ==================================================

elif st.session_state.page == "AI Engine":

    st.title("AI Prediction Engine")

    symbol = st.text_input("Enter Stock Symbol", "AAPL")

    data = yf.download(symbol, period="2y")

    if data.empty:
        st.error("Invalid stock symbol.")
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

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    st.metric("Model Accuracy", f"{round(accuracy*100,2)}%")

    latest = X.iloc[-1:].values
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0]

    if pred == 1:
        st.success("Model Signal: BUY")
        st.metric("Confidence Level", f"{round(prob[1]*100,2)}%")
    else:
        st.error("Model Signal: SELL")
        st.metric("Confidence Level", f"{round(prob[0]*100,2)}%")

# ==================================================
# BACKTESTING
# ==================================================

elif st.session_state.page == "Backtesting Laboratory":

    st.title("Strategy Backtesting Laboratory")

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
    ax.plot(test["Market"], label="Buy and Hold")
    ax.plot(test["AI"], label="AI Strategy")
    ax.legend()
    st.pyplot(fig)

# ==================================================
# ABOUT
# ==================================================

elif st.session_state.page == "About Us":

    st.title("About QuantNova")

    st.write("""
    QuantNova is an academic artificial intelligence research initiative
    developed to demonstrate the integration of machine learning with
    financial market analytics. The project aims to bridge theoretical
    data science principles with real-world quantitative trading concepts.
    """)

# ==================================================
# CONTACT
# ==================================================

elif st.session_state.page == "Contact":

    st.title("Contact")

    st.write("Email: quantnova.ai@gmail.com")
    st.write("Location: Academic Research Initiative")

# ==================================================
# FOLLOW
# ==================================================

elif st.session_state.page == "Follow Us":

    st.title("Follow")

    st.write("LinkedIn: linkedin.com/company/quantnova")
    st.write("Twitter: twitter.com/quantnova_ai")
    st.write("Instagram: instagram.com/quantnova_ai")

# ==================================================
# COPYRIGHT
# ==================================================

st.markdown("---")
st.markdown(f"© {datetime.datetime.now().year} QuantNova AI Research Lab")
