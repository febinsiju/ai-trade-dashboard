import streamlit as st

st.set_page_config(
    page_title="AI Trading Intelligence Platform",
    layout="wide",
    page_icon="📊"
)

st.markdown("""
<style>
body { background-color: #0E1117; }
.main { background-color: #0E1117; }
.block-container { padding-top: 2rem; }

.hero-title {
    font-size: 50px;
    font-weight: 800;
    color: #00FFA3;
}

.section-box {
    background-color: #141A2A;
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">AI Trading Intelligence Platform</div>', unsafe_allow_html=True)

st.write("""
This platform is designed to deliver institutional-grade trading research tools.

We combine machine learning, quantitative analytics, and technical indicators
to help traders make structured, data-driven decisions.

Our system enables:

• AI-powered trade prediction  
• Historical backtesting  
• Risk-adjusted performance analytics  
• Real-time market monitoring  

Select a module below to begin.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("📈 AI Prediction Engine")
    st.write("Generate BUY/SELL signals using machine learning models.")
    st.page_link("pages/1_Predictor.py", label="Launch Prediction Engine →")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("🧪 Strategy Backtesting Lab")
    st.write("Evaluate model performance vs Buy & Hold strategies.")
    st.page_link("pages/2_Backtest.py", label="Launch Backtesting Lab →")
    st.markdown('</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("📊 Risk Analytics Dashboard")
    st.write("Analyze Sharpe ratio, drawdowns, volatility and risk exposure.")
    st.page_link("pages/3_Risk_Analytics.py", label="Open Risk Dashboard →")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("🌍 Market Overview")
    st.write("View global markets and technical charts.")
    st.page_link("pages/4_Market_Dashboard.py", label="Open Market Dashboard →")
    st.markdown('</div>', unsafe_allow_html=True)
