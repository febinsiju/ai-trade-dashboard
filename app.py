import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import base64
from io import BytesIO
import os
import datetime

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="QuantNova", layout="wide")

# =====================================================
# DARK SaaS STYLE
# =====================================================

st.markdown("""
<style>

/* GLOBAL BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #0E1117 0%, #111827 50%, #0B0F1A 100%);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* FADE IN ANIMATION */
.fade-in {
    animation: fadeIn 1.5s ease-in;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0px);}
}

/* GLOW TITLE */
.glow-text {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00C6FF, #0072FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* PREMIUM BUTTON */
.stButton > button {
    background: linear-gradient(45deg, #0072FF, #00C6FF);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton > button:hover {
    box-shadow: 0 0 20px #00C6FF;
    transform: scale(1.05);
}

/* CARD EFFECT */
.card {
    background: rgba(255, 255, 255, 0.03);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 0 25px rgba(0, 198, 255, 0.3);
}

/* METRIC STYLE */
[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: 700;
    color: #00C6FF;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #0B0F1A;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

# =====================================================
# IMAGE FUNCTION (REQUIRED FOR ABOUT PAGE)
# =====================================================

def circular_image(image_path, size=180):
    if not os.path.exists(image_path):
        st.warning(f"{image_path} not found.")
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
# SIDEBAR (SAFE NAVIGATION)
# =====================================================

st.sidebar.title("QuantNova Platform")

pages = ["Home", "AI Intelligence Engine", "Strategy Lab", "Market Dashboard"]

if st.session_state.page in pages:
    idx = pages.index(st.session_state.page)
else:
    idx = 0

selected = st.sidebar.radio("Navigate", pages, index=idx)

# Only update if current page is a sidebar page
if selected != st.session_state.page and st.session_state.page in pages:
    st.session_state.page = selected

# =====================================================
# HOME
# =====================================================

if st.session_state.page == "Home":

    st.markdown('<div class="fade-in glow-text">QuantNova</div>', unsafe_allow_html=True)
    st.markdown('<div class="fade-in" style="font-size:1.5rem; font-weight:500;">AI-Powered Quantitative Intelligence Platform</div>', unsafe_allow_html=True)

    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.markdown("""The Operating System for AI-Driven Market Intelligence

QuantNova is not entering the financial analytics industry.
It is architecting the layer that will sit beneath it.

We are building an AI-native quantitative intelligence operating system designed to convert global market complexity into structured, self-evolving, probabilistic decision architecture.

Financial markets generate terabytes of data every second — but raw data is noise without structured intelligence. The next era of dominance will not belong to those who see more data. It will belong to those who can model uncertainty, quantify asymmetry, validate structure, and adapt faster than systemic change.

QuantNova is engineered for that era.

At its foundation lies a multi-layer intelligence stack:

• Adaptive ensemble learning systems
• Probabilistic modeling and uncertainty quantification
• Statistical validation and structural integrity engines
• Modular experimentation frameworks
• Scalable backtesting and performance analytics cores
• Infrastructure-ready deployment architecture

Every signal is measurable.
Every prediction is probabilistic.
Every model is reproducible.
Every system is expandable.

QuantNova is not a dashboard.
It is not a signal bot.
It is not a retail trading assistant.

It is a scalable intelligence infrastructure capable of evolving into:

• Cross-asset AI prediction networks
• Institutional-grade strategy simulation ecosystems
• Autonomous model evolution engines
• High-frequency data structuring pipelines
• Enterprise API intelligence layers
• AI-powered hedge fund architecture

Markets are increasingly algorithmic. Capital is increasingly automated. Decision cycles are increasingly compressed.

The companies that win in this environment will not build tools — they will build infrastructure.

QuantNova is being designed as that infrastructure.

Not to compete for attention.
But to become foundational.

Where others optimize indicators, we engineer intelligence systems.
Where others search for signals, we construct predictive architecture.
Where others iterate features, we architect dominance.

QuantNova is not a startup experimenting with finance.
It is a research-driven AI systems company entering financial intelligence as its first domain of deployment.""")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    if st.button("About Us"):
        st.session_state.page = "About"
        st.rerun()

    st.markdown("---")
    st.header("Contact Us")
    st.write("+91 8089411348")
    st.write("+91 7012958445")

    st.header("Follow Us On")
    st.write("@f_eb_in_")
    st.write("@_gan.ga__")

# =====================================================
# AI ENGINE
# =====================================================

elif st.session_state.page == "AI Intelligence Engine":

    st.title("AI Intelligence Engine")

    symbol = st.text_input("Stock Symbol", "AAPL")
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    prob = model.predict_proba(X_test.tail(1))[0]

    prediction = "BUY" if prob[1] > prob[0] else "SELL"
    confidence = round(max(prob) * 100, 2)

    col1, col2 = st.columns(2)
    col1.metric("Model Accuracy", f"{round(accuracy*100,2)}%")
    col2.metric("Signal", f"{prediction} ({confidence}% confidence)")

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, model.predict(X_test)))

# =====================================================
# STRATEGY LAB
# =====================================================

elif st.session_state.page == "Strategy Lab":

    st.title("Strategy Lab")

    symbol = st.text_input("Stock Symbol", "AAPL")
    data = yf.download(symbol, period="2y")

    short = st.slider("Short SMA", 5, 30, 10)
    long = st.slider("Long SMA", 20, 100, 50)

    data["Short"] = data["Close"].rolling(short).mean()
    data["Long"] = data["Close"].rolling(long).mean()
    data["Signal"] = np.where(data["Short"] > data["Long"], 1, 0)
    data["Returns"] = data["Close"].pct_change()
    data["Strategy"] = data["Signal"].shift(1) * data["Returns"]

    cumulative = (1 + data["Strategy"]).cumprod()

    st.line_chart(cumulative)

    total_return = round((cumulative.iloc[-1] - 1) * 100, 2)
    st.metric("Total Strategy Return", f"{total_return}%")

# =====================================================
# MARKET DASHBOARD
# =====================================================

elif st.session_state.page == "Market Dashboard":

    st.title("Market Dashboard")

    symbol = st.text_input("Stock Symbol", "AAPL")
    data = yf.download(symbol, period="6mo")

    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])

    fig.update_layout(
    template="plotly_dark",
    transition_duration=500
)
st.plotly_chart(fig, use_container_width=True)

# =====================================================
# FULL ABOUT PAGE (UNCHANGED)
# =====================================================

    elif st.session_state.page == "About":

    st.title("About QuantNova")

    st.markdown("""
    QuantNova was conceived as an academic research initiative by students of
    Computer Science and Engineering (CSE B S2) at TocH Institute Of Science And Technology (TIST),
    Ernakulam, Kerala.

    The project reflects an ambition to bridge theoretical machine learning concepts
    with practical financial data modeling. It represents a structured effort to build,
    test, validate, and continuously refine predictive intelligence systems within a
    disciplined research framework.
    """)

    st.markdown("---")

    circular_image("founder_image.jpg", 180)
    st.markdown("<h3 style='text-align:center;'>Febin Siju</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-weight:600;'>Founder & Lead Architect</p>", unsafe_allow_html=True)

    st.markdown("""
    Febin Siju serves as the Founder and Lead Architect of QuantNova. He led the
    conceptual design and technical implementation of the platform’s AI framework,
    including feature engineering pipelines, ensemble modeling architecture, and
    systematic validation procedures.

    His focus lies in structured experimentation, ensuring that predictive outputs
    are grounded in statistical reasoning rather than assumption. By integrating
    financial time-series analysis with machine learning algorithms, he aimed to
    construct a system capable of evolving through iterative exposure to market data.

    Beyond coding implementation, Febin directed the architectural blueprint of the
    platform — defining modular components, ensuring data preprocessing integrity,
    and aligning the research objectives with academic rigor.

    His long-term vision for QuantNova involves expanding toward adaptive learning
    frameworks and advanced predictive calibration methodologies.
    """)

    st.markdown("---")

    circular_image("ganga_image.jpg", 180)
    st.markdown("<h3 style='text-align:center;'>Ganga AR</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-weight:600;'>Co-Founder & Research Strategist</p>", unsafe_allow_html=True)

    st.markdown("""
    Ganga AR serves as Co-Founder and Research Strategist for QuantNova. Her
    contributions focused on analytical validation, structured evaluation of
    model outputs, and refinement of documentation standards to ensure clarity
    and academic integrity.

    Her involvement ensured that QuantNova remained not only a functional
    software system but also a well-documented research framework capable
    of academic presentation and further development.
    """)
    st.markdown("---")
    if st.button("Back to Home"):
        st.session_state.page = "Home"
        st.rerun()

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown(f"© {datetime.datetime.now().year} QuantNova | SaaS Research Build v1.0")
