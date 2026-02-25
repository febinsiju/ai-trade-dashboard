import streamlit as st

st.set_page_config(
    page_title="AI Trading Intelligence Platform",
    layout="wide",
    page_icon="📊"
)

# =========================
# PROFESSIONAL STYLING
# =========================
st.markdown("""
<style>
body { background-color: #0E1117; }
.main { background-color: #0E1117; }
.block-container { padding-top: 2rem; }

.big-title {
    font-size: 48px;
    font-weight: 700;
    color: #00FFA3;
}

.section-box {
    background-color: #141A2A;
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HERO SECTION
# =========================
st.markdown('<div class="big-title">AI Trading Intelligence Platform</div>', unsafe_allow_html=True)

st.write("""
Welcome to the next-generation AI-powered trading research terminal.

This platform is designed to analyze financial markets using machine learning models,
technical indicators, and quantitative risk frameworks.

Our goal is not just prediction — but intelligent decision support.

We combine:
• Market data from live sources  
• AI-driven signal generation  
• Risk-adjusted backtesting  
• Institutional-level performance metrics  

Explore the modules below to begin your analysis.
""")

st.divider()

# =========================
# FEATURE SECTIONS
# =========================

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("📈 AI Prediction Engine")
    st.write("""
Analyze any stock or cryptocurrency using machine learning.
Generate BUY or SELL signals based on quantitative models.
    """)
    st.page_link("pages/1_Predictor.py", label="Launch Prediction Engine →")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("🧪 Strategy Backtesting")
    st.write("""
Test your AI strategy against historical data.
Measure performance vs Buy & Hold.
    """)
    st.page_link("pages/2_Backtest.py", label="Launch Backtesting →")
    st.markdown('</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("📊 Risk Analytics")
    st.write("""
Evaluate Sharpe Ratio, drawdowns, volatility,
and institutional-grade performance metrics.
    """)
    st.page_link("pages/3_Risk_Analytics.py", label="Open Risk Dashboard →")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("🌍 Market Dashboard")
    st.write("""
View global markets, crypto performance,
and technical indicator panels in real time.
    """)
    st.page_link("pages/4_Market_Dashboard.py", label="Open Market Dashboard →")
    st.markdown('</div>', unsafe_allow_html=True)
