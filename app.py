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
body {background-color: #0E1117; color: white;}
.block-container {padding-top: 2rem;}
h1, h2, h3 {color: #FFFFFF;}
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

    st.title("QuantNova")
    st.subheader("AI-Powered Quantitative Intelligence Platform")

    st.markdown("""
QuantNova is a next-generation research-driven AI platform designed to transform financial market data into structured predictive intelligence.

Built with a startup mindset and academic discipline, the system integrates ensemble learning, statistical validation, and modular experimentation frameworks to deliver analytical clarity — not speculation.
""")

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

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ABOUT PAGE (HOVER VERSION - FIXED INDENTATION)
# =====================================================

elif st.session_state.page == "About":

    st.title("About QuantNova")

    st.markdown("""
QuantNova was conceived as a long-horizon academic research initiative by the members of Group 7, students of Computer Science and Engineering (CSE B S2) at TocH Institute Of Science And Technology (TIST), Ernakulam, Kerala.

As Group 7, our aim extends far beyond academic submission. We are focused on building structured, research-oriented artificial intelligence systems that combine statistical rigor, engineering discipline, and scalable system architecture. Our mission is to transform theoretical knowledge into practical, measurable intelligence frameworks capable of operating in complex financial ecosystems.

We believe markets are probabilistic systems, not deterministic machines. Through QuantNova, our objective is to design modeling frameworks that quantify uncertainty, validate predictive structures, and evolve through disciplined experimentation cycles.

QuantNova represents our commitment to engineering infrastructure — not just features — and to approaching AI development with both academic integrity and startup ambition.
""")

    st.markdown("---")

    # =============================
    # HOVER STYLE (ONLY ONCE)
    # =============================
    st.markdown("""
<style>
.profile-container {
    position: relative;
    width: 220px;
    height: 220px;
    margin: 30px auto;
}

.profile-image {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    object-fit: cover;      /* 🔥 prevents stretching */
    object-position: center;/* 🔥 centers the face */
    display: block;
    transition: 0.4s ease;
}

.profile-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border-radius: 50%;
    background: rgba(0, 0, 0, 0.85);
    color: white;
    opacity: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 15px;
    font-size: 14px;
    transition: 0.4s ease;
}

.profile-container:hover .profile-overlay {
    opacity: 1;
}

.profile-container:hover .profile-image {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

    # =============================
    # IMAGE ENCODING FUNCTION
    # =============================
    def get_base64_image(path):
        if not os.path.exists(path):
            return ""
        with open(path, "rb") as img:
            return base64.b64encode(img.read()).decode()

    # =============================
    # FOUNDER
    # =============================
    founder_img = get_base64_image("founder_image.jpg")

    st.markdown(f"""
    <div class="profile-container">
        <img src="data:image/png;base64,{founder_img}" class="profile-image">
        <div class="profile-overlay">
            Founder & Lead Architect of QuantNova.<br>
            Designed AI architecture, predictive systems, 
            and long-term intelligence infrastructure roadmap.
        </div>
    </div>

    <h3 style="text-align:center;">Febin Siju</h3>
    <p style="text-align:center; font-weight:600;">Founder & Lead Architect</p>
    """, unsafe_allow_html=True)

    st.markdown("""
Febin Siju architected QuantNova from its foundational system design to its advanced modeling logic. His work integrates ensemble learning systems, statistical validation processes, and modular intelligence frameworks into a unified predictive architecture. His focus is long-term scalability, structural clarity, and measurable AI performance.
""")

    st.markdown("---")

    # =============================
    # CO-FOUNDER
    # =============================
    cofounder_img = get_base64_image("ganga_image.jpg")

    st.markdown(f"""
    <div class="profile-container">
        <img src="data:image/png;base64,{cofounder_img}" class="profile-image">
        <div class="profile-overlay">
            Co-Founder & Research Strategist.<br>
            Leads validation methodology, structured experimentation, 
            and analytical integrity.
        </div>
    </div>

    <h3 style="text-align:center;">Ganga AR</h3>
    <p style="text-align:center; font-weight:600;">Co-Founder & Research Strategist</p>
    """, unsafe_allow_html=True)

    st.markdown("""
Ganga AR strengthens QuantNova’s research discipline through structured validation frameworks, reproducible experimentation processes, and rigorous analytical documentation. Her focus ensures the platform maintains academic integrity while evolving toward scalable AI intelligence infrastructure.
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
