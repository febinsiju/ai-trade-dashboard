import streamlit as st

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="QuantNova", layout="wide")

# =====================================================
# THEME TOGGLE
# =====================================================
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

theme_choice = st.sidebar.selectbox("Select Theme", ["Dark", "Light"])

if theme_choice == "Dark":
    bg_color = "#0f172a"
    card_color = "#1e293b"
    text_color = "white"
else:
    bg_color = "#f8fafc"
    card_color = "white"
    text_color = "black"

# =====================================================
# GLOBAL STYLE
# =====================================================
st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: {text_color};
}}

.hero {{
    text-align: center;
    padding: 60px 20px;
}}

.hero h1 {{
    font-size: 48px;
}}

.card {{
    background: {card_color};
    padding: 25px;
    border-radius: 15px;
    margin: 15px 0;
    box-shadow: 0px 0px 25px rgba(56,189,248,0.25);
}}

.big-section {{
    padding: 40px 0px;
}}
</style>
""", unsafe_allow_html=True)

# =====================================================
# NAVIGATION
# =====================================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("Home"):
        st.session_state.page = "Home"
with col2:
    if st.button("AI Trade Bot"):
        st.session_state.page = "AI Trade Bot"
with col3:
    if st.button("Strategy Lab"):
        st.session_state.page = "Strategy Lab"
with col4:
    if st.button("Market Dashboard"):
        st.session_state.page = "Market Dashboard"
with col5:
    if st.button("About Us"):
        st.session_state.page = "About"

st.markdown("---")

# =====================================================
# HOME PAGE
# =====================================================
if st.session_state.page == "Home":

    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown("<h1>QuantNova</h1>", unsafe_allow_html=True)
    st.markdown("### AI-Powered Quantitative Intelligence Platform")
    st.markdown("""
QuantNova is an advanced AI trading ecosystem designed to convert market complexity into structured decision intelligence.

By integrating adaptive machine learning systems, statistical validation engines, and dynamic strategy modeling frameworks, QuantNova enables traders and researchers to operate with institutional-grade analytical precision.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.header("Platform Overview")

    st.markdown("""
QuantNova is structured around four core pillars: AI Trade Bots, Strategy Lab, Market Intelligence Dashboard, and Research Infrastructure. 

The platform is engineered not as a signal provider, but as a predictive intelligence system built for disciplined quantitative execution.
""")

# =====================================================
# AI TRADE BOT PAGE
# =====================================================
elif st.session_state.page == "AI Trade Bot":

    st.title("AI Trade Bot Marketplace")

    st.markdown("""
Explore autonomous AI trading systems built on volatility modeling, momentum detection, and structural market analytics.
""")

    bots = [
        {"name": "Momentum Alpha Bot", "return": "+214%", "desc": "Short-term adaptive momentum strategy."},
        {"name": "TrendNova Engine", "return": "+178%", "desc": "Multi-timeframe trend detection AI."},
        {"name": "Quantum Hedge AI", "return": "+196%", "desc": "Volatility-optimized hedge framework."}
    ]

    for bot in bots:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(bot["name"])
        st.write(f"Annualized Return: {bot['return']}")
        st.write(bot["desc"])
        st.button(f"View Details - {bot['name']}")
        st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# STRATEGY LAB PAGE
# =====================================================
elif st.session_state.page == "Strategy Lab":

    st.title("Strategy Lab")

    st.markdown("""
The Strategy Lab is the experimental and validation environment of QuantNova.

Here, strategies undergo walk-forward optimization, Monte Carlo simulation, regime stress testing, and performance attribution analysis.

Every trading framework deployed through QuantNova is validated across multiple volatility conditions to ensure structural robustness rather than curve-fitted illusion.
""")

# =====================================================
# MARKET DASHBOARD PAGE
# =====================================================
elif st.session_state.page == "Market Dashboard":

    st.title("Market Intelligence Dashboard")

    st.markdown("""
The Market Dashboard provides structured macro and micro market awareness.

It integrates volatility indices, liquidity diagnostics, sector rotation models, and risk exposure mapping into a unified decision interface.

Designed for clarity over noise, the dashboard highlights only statistically relevant signals that align with prevailing market regimes.
""")

# =====================================================
# ABOUT PAGE (UNCHANGED STRUCTURE)
# =====================================================
elif st.session_state.page == "About":

    st.title("About Us")

    st.markdown("""
QuantNova is a research-driven AI initiative dedicated to advancing quantitative trading intelligence.

Built with a foundation in data science, financial mathematics, and adaptive systems engineering, the platform aims to redefine how individuals and institutions interact with financial markets.
""")

    st.markdown("---")

    st.header("Contact Us")
    st.write("+91 8089411348")
    st.write("+91 7012958445")

    st.header("Follow Us On")
    st.write("@f_eb_in_")
    st.write("@_gan.ga__")
