import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime
import os

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="QuantNova AI Trading Intelligence",
    layout="wide"
)

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    ["Home", "AI Engine", "Backtesting Laboratory", "About Us", "Contact", "Follow Us"]
)

st.session_state.page = page

# =====================================================
# GLOBAL STYLING
# =====================================================

st.markdown("""
<style>

.big-title {
    font-size: 52px;
    font-weight: 800;
    margin-bottom: 30px;
}

.section-heading {
    font-size: 38px;
    font-weight: 700;
    margin-top: 70px;
    margin-bottom: 25px;
}

.large-paragraph {
    font-size: 20px;
    line-height: 1.9;
    text-align: justify;
    margin-bottom: 30px;
}

.profile-card {
    background-color: #ffffff;
    padding: 40px;
    border-radius: 12px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
    margin-bottom: 60px;
}

.profile-name {
    font-size: 26px;
    font-weight: 700;
    margin-top: 10px;
    text-align: center;
}

.profile-role {
    font-size: 18px;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 20px;
    text-align: center;
}

.profile-text {
    font-size: 18px;
    line-height: 1.8;
    text-align: justify;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HOME
# =====================================================

if st.session_state.page == "Home":

    st.markdown('<div class="big-title">QuantNova AI Trading Intelligence Platform</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="large-paragraph">
    QuantNova is a quantitative research platform designed to integrate artificial
    intelligence with financial market analytics. The system transforms historical
    price data into structured predictive intelligence using disciplined machine
    learning methodologies and statistical reasoning.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="large-paragraph">
    By applying ensemble learning algorithms and structured validation pipelines,
    the platform generates probabilistic signals to assist in evaluating potential
    market direction. Emphasis is placed on risk-aware interpretation rather than
    speculative certainty.
    </div>
    """, unsafe_allow_html=True)

    st.info("Developed strictly for academic research and demonstration purposes.")

# =====================================================
# AI ENGINE
# =====================================================

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

    accuracy = accuracy_score(y_test, model.predict(X_test))
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

# =====================================================
# BACKTESTING
# =====================================================

elif st.session_state.page == "Backtesting Laboratory":

    st.title("Strategy Backtesting Laboratory")

    symbol = st.text_input("Stock Symbol", "AAPL")

    data = yf.download(symbol, period="2y")
    data["Return"] = data["Close"].pct_change()
    data = data.dropna()

    data["Market"] = (1 + data["Return"]).cumprod()

    fig, ax = plt.subplots()
    ax.plot(data["Market"], label="Buy and Hold")
    ax.legend()
    st.pyplot(fig)

# =====================================================
# ABOUT US
# =====================================================

elif st.session_state.page == "About Us":

    st.title("About QuantNova")

    st.markdown("""
    We are CSE B S2 students of TocH Institute Of Science And Technology (TIST),
    Ernakulam, Kerala. QuantNova was developed as an academic initiative to explore
    the practical application of artificial intelligence in financial prediction systems.
    """)

    # Founder Card
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)

    if os.path.exists("founder_image.jpg"):
        st.image("founder_image.jpg", width=160)
    else:
        st.warning("Founder image not found")
        st.markdown("""
        <div style="text-align:center;">
            <img src="founder_image.jpg" width="160"
            style="border-radius:50%; margin-bottom:20px;">
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="profile-name">[Your Name]</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-role">Founder & Lead Architect</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="profile-text">
    Architect of the AI framework, predictive modeling system, and validation pipeline.
    Led the conceptualization and implementation of QuantNova with the objective of
    creating a structured machine learning system capable of evolving through data exposure.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Co-Founder Card
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)

    if os.path.exists("ganga_image.jpg"):
        st.markdown("""
        <div style="text-align:center;">
            <img src="ganga_image.jpg" width="160"
            style="border-radius:50%; margin-bottom:20px;">
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="profile-name">Ganga AR</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-role">Co-Founder & Research Strategist</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="profile-text">
    Contributed to analytical validation, performance evaluation, and structured
    documentation refinement. Played a key role in strengthening the academic
    and research foundations of the platform.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# CONTACT
# =====================================================

elif st.session_state.page == "Contact":
    st.title("Contact")
    st.write("Email: quantnova.ai@gmail.com")
    st.write("Institution: TocH Institute Of Science And Technology")

# =====================================================
# FOLLOW
# =====================================================

elif st.session_state.page == "Follow Us":
    st.title("Follow Us")
    st.write("LinkedIn")
    st.write("Instagram")
    st.write("Twitter")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown(f"© {datetime.datetime.now().year} QuantNova AI Research Initiative")
