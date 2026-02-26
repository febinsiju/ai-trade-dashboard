import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import base64
from io import BytesIO
import os
import datetime

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="QuantNova AI Trading Intelligence",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

# =====================================================
# IMAGE FUNCTION
# =====================================================

def circular_image(image_path, size=180):
    if not os.path.exists(image_path):
        st.warning(f"{image_path} not found in project folder.")
        return

    img = Image.open(image_path)

    width, height = img.size
    min_dim = min(width, height)

    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2

    img = img.crop((left, top, right, bottom))
    img = img.resize((size, size))

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()

    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/png;base64,{encoded}"
                 style="border-radius:50%;
                        width:{size}px;
                        height:{size}px;
                        object-fit:cover;">
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# SIDEBAR (Only Core Navigation)
# =====================================================

st.sidebar.title("Navigation")

sidebar_options = ["Home", "AI Engine", "Backtesting Laboratory"]

if st.session_state.page in sidebar_options:
    current_index = sidebar_options.index(st.session_state.page)
else:
    current_index = 0

selected = st.sidebar.radio(
    "Select Section",
    sidebar_options,
    index=current_index
)

if selected != st.session_state.page and st.session_state.page in sidebar_options:
    st.session_state.page = selected

# =====================================================
# HOME PAGE
# =====================================================

if st.session_state.page == "Home":

    st.title("QuantNova AI Trading Intelligence Platform")

    st.markdown("""
    QuantNova represents a structured quantitative research initiative developed to explore the integration of artificial intelligence and financial market modeling. The platform is designed to analyze historical time-series data, extract meaningful statistical patterns, and generate predictive insights using ensemble learning techniques.

    This system is not built for speculative hype or superficial prediction. Instead, it is constructed with a research-oriented mindset that prioritizes validation, interpretability, and systematic experimentation. Every modeling decision is made with academic discipline and statistical reasoning.

    By combining financial indicators with machine learning classifiers, QuantNova transforms raw data into structured decision-support intelligence. The long-term objective is to evolve the system into a robust research-grade predictive framework capable of adaptive learning and rigorous backtesting.
    """)

    st.info("Developed strictly for academic research and demonstration purposes.")

    st.markdown("---")
    st.markdown("### Learn More")

    col1 = st.columns(1)[0]

    with col1:
        if st.button("About Us"):
            st.session_state.page = "About"
            st.rerun()

    # =====================================================
    # CONTACT & FOLLOW SECTION (NOT BUTTONS)
    # =====================================================

    st.markdown("---")
    st.markdown("## Contact Us")
    st.markdown("""
    +91 8089411348  
    +91 7012958445
    """)

    st.markdown("## Follow Us On")
    st.markdown("""
    @f_eb_in_  
    @_gan.ga__
    """)

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

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    st.metric("Model Accuracy", f"{round(accuracy*100,2)}%")

# =====================================================
# ABOUT PAGE (UNCHANGED)
# =====================================================

elif st.session_state.page == "About":

    st.title("About QuantNova")

    st.markdown("""
    QuantNova was conceived as an academic research initiative by students of Computer Science and Engineering (CSE B S2) at TocH Institute Of Science And Technology (TIST), Ernakulam, Kerala. The project represents a disciplined effort to translate theoretical machine learning knowledge into a structured financial analytics system.

    The initiative was built with a long-term vision of merging statistical rigor, algorithmic experimentation, and responsible AI modeling into a unified research platform. Rather than creating a simple dashboard, the objective was to architect a scalable framework capable of evolving through iterative experimentation.
    """)

    st.markdown("---")

    circular_image("founder_image.jpg", 180)
    st.markdown("<h3 style='text-align:center;'>Febin Siju</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-weight:600;'>Founder & Lead Architect</p>", unsafe_allow_html=True)

    st.markdown("""
    Febin Siju conceptualized and architected QuantNova from its foundational framework to its advanced modeling components. His responsibilities encompassed system architecture design, feature engineering logic, model experimentation, and validation strategy development.

    His focus was on building a modular and research-aligned structure capable of sustaining future expansion. By integrating ensemble-based classification models and carefully engineered financial indicators, he established the predictive backbone of the platform.

    Beyond implementation, his aim was to cultivate a disciplined research culture — where results are interpreted cautiously, validated rigorously, and continuously refined.
    """)

    st.markdown("---")

    circular_image("ganga_image.jpg", 180)
    st.markdown("<h3 style='text-align:center;'>Ganga AR</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-weight:600;'>Co-Founder & Research Strategist</p>", unsafe_allow_html=True)

    st.markdown("""
    Ganga AR played a crucial role in refining the analytical integrity of QuantNova. Her contributions centered on validation methodology, structured documentation, and interpretative clarity of predictive results.

    She ensured that the system remained aligned with academic standards, emphasizing transparency in model behavior and clarity in performance reporting.
    """)

    st.markdown("---")
    if st.button("Back to Home"):
        st.session_state.page = "Home"
        st.rerun()

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown(f"© {datetime.datetime.now().year} QuantNova AI Research Initiative")
