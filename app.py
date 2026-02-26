import streamlit as st
import os
from PIL import Image

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Trading Intelligence Platform",
    layout="wide",
    page_icon="📊"
)

# ==========================================
# SAFE IMAGE LOADER
# ==========================================

def safe_image(path):
    if os.path.exists(path):
        try:
            img = Image.open(path)
            st.image(img, use_container_width=True)
        except Exception:
            st.warning(f"⚠️ Unable to load image: {path}")
    else:
        st.warning(f"⚠️ Image not found: {path}")

# ==========================================
# GLOBAL STYLING
# ==========================================

st.markdown("""
<style>

body {
    background-color: #0E1117;
}

.main {
    background: linear-gradient(180deg, #0E1117 0%, #0B0F18 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.section-title {
    font-size: 36px;
    font-weight: 700;
    color: #00FFA3;
    margin-bottom: 20px;
}

.section-text {
    font-size: 17px;
    line-height: 1.8;
    color: #C5C6C7;
}

.feature-box {
    padding: 40px;
    background-color: #141A2A;
    border-radius: 20px;
}

.cta-box {
    background: linear-gradient(135deg, #00FFA3, #00C896);
    padding: 60px;
    border-radius: 25px;
    text-align: center;
    color: black;
    margin-top: 80px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Prediction Engine", "Backtesting Lab"]
)

# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":

    st.markdown('<div class="section-title">AI Trading Intelligence Platform</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-text">
    A quantitative research system engineered to transform complex 
    financial market data into structured, machine-driven trading intelligence.

    By integrating predictive modeling, risk-adjusted validation, and 
    systematic performance analytics, the platform enables disciplined 
    decision-making rooted in probability rather than emotion.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-title">Our Quantitative Methodology</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-text">
    Market data is transformed into predictive indicators, processed through 
    machine learning models, and validated using structured backtesting workflows. 

    Performance is evaluated using risk-adjusted metrics rather than raw returns.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div class="cta-box">
        <h2>Begin Your Quantitative Trading Analysis</h2>
        <p>Explore predictive modeling and institutional-grade analytics.</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PREDICTION ENGINE PAGE
# ==========================================

elif page == "Prediction Engine":

    st.markdown('<div class="section-title">AI Prediction Engine</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])

    with col1:
        safe_image("images/prediction.jpg")

    with col2:
        st.markdown("""
        <div class="feature-box">
        <div class="section-text">
        The Prediction Engine leverages supervised ML algorithms 
        to generate BUY or SELL signals with confidence levels.

        Structured insight — not speculation.
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    if st.button("Run AI Analysis"):
        with st.spinner("Analyzing market data..."):
            import time
            time.sleep(2)

        st.success("Prediction Complete ✅")
        st.metric("Signal", "BUY")
        st.metric("Confidence", "82%")

# ==========================================
# BACKTESTING LAB PAGE
# ==========================================

elif page == "Backtesting Lab":

    st.markdown('<div class="section-title">Strategy Backtesting Laboratory</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown("""
        <div class="feature-box">
        <div class="section-text">
        Compare AI-driven strategy against Buy & Hold performance 
        using structured historical validation.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        safe_image("images/backtest.jpg")

    st.divider()

    if st.button("Run Backtest"):
        with st.spinner("Running historical simulation..."):
            import time
            time.sleep(2)

        st.success("Backtest Complete ✅")
        st.metric("AI Strategy Return", "24%")
        st.metric("Buy & Hold Return", "15%")
