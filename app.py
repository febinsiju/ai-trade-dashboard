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
# ABOUT PAGE (EXPANDED VERSION)
# =====================================================

elif st.session_state.page == "About":

    st.title("About QuantNova")

    st.markdown("""
QuantNova was conceived as a long-horizon academic research initiative by the members of **Group 7**, students of Computer Science and Engineering (CSE B S2) at TocH Institute Of Science And Technology (TIST), Ernakulam, Kerala. 

What began as a structured academic exploration into financial data modeling gradually evolved into a disciplined research mission focused on bridging the gap between theoretical artificial intelligence frameworks and their real-world deployment in financial intelligence systems. As members of Group 7, our objective was never limited to building a classroom project. Instead, we sought to architect a foundational system that demonstrates how statistical reasoning, computational rigor, and algorithmic experimentation can converge into a scalable intelligence platform.

Our collective aim is to design systems that do not merely react to market movements but interpret structural patterns within them. We believe financial markets represent one of the most complex real-time data ecosystems in existence — continuously evolving, probabilistic in nature, and heavily influenced by behavioral, economic, and technological forces. QuantNova is our attempt to engineer clarity within that complexity.

As a research-driven team, our mission extends beyond feature implementation. We focus on validation methodology, reproducibility of experiments, measurable uncertainty modeling, and structured system design. Group 7 operates with a philosophy rooted in long-term intellectual discipline rather than short-term experimentation. Our goal is to cultivate a platform that demonstrates how academic precision can coexist with startup-level ambition.

QuantNova represents our commitment to building systems that prioritize integrity over hype, structure over speculation, and measurable performance over assumption. It is a living research framework — one that we intend to continuously expand, refine, and elevate into a platform capable of evolving alongside advancements in machine learning and financial modeling.

We are not simply students completing a requirement. We are engineers building infrastructure. We are researchers designing frameworks. And we are founders shaping a vision that extends far beyond a single semester or deployment cycle.
""")

    st.markdown("---")

    circular_image("founder_image.jpg", 180)
    st.markdown("<h3 style='text-align:center;'>Febin Siju</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-weight:600;'>Founder & Lead Architect</p>", unsafe_allow_html=True)

    st.markdown("""
Febin Siju is the founding architect of QuantNova and the principal systems thinker behind its structural design. From conceptualization to execution, he engineered the foundational framework that defines the platform’s analytical depth and operational coherence.

His approach to system design is rooted in long-term architectural thinking. Rather than assembling isolated machine learning components, Febin structured QuantNova as a layered intelligence ecosystem. He focused on modular design principles, enabling experimentation without compromising structural stability. The predictive modeling pipelines, feature engineering frameworks, and validation strategies were conceptualized with scalability in mind — ensuring that each component could evolve independently while remaining integrated within a cohesive architecture.

Febin’s technical focus spans ensemble learning systems, probabilistic outcome modeling, statistical validation mechanics, and deployment-oriented system thinking. He emphasizes the importance of measurable intelligence — where every signal must be reproducible, every experiment must be documented, and every prediction must be statistically defensible. His philosophy rejects black-box assumptions and instead promotes transparent, testable modeling processes.

Beyond technical implementation, Febin established the long-term strategic vision of QuantNova. He positioned the platform not merely as an academic submission but as a scalable research infrastructure capable of future expansion into cross-asset intelligence modeling, institutional-grade strategy validation systems, and enterprise-ready AI analytics layers.

His leadership philosophy integrates academic discipline with startup momentum. He believes that innovation is not achieved through rapid feature accumulation, but through carefully engineered structural foundations that can withstand complexity over time. QuantNova reflects this mindset — deliberate, research-oriented, and architecturally intentional.

Through disciplined experimentation and structured iteration, Febin continues to guide QuantNova’s evolution toward becoming a sophisticated AI-driven quantitative intelligence framework capable of adapting to increasingly complex financial ecosystems.
""")

    st.markdown("---")

    circular_image("ganga_image.jpg", 180)
    st.markdown("<h3 style='text-align:center;'>Ganga AR</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-weight:600;'>Co-Founder & Research Strategist</p>", unsafe_allow_html=True)

    st.markdown("""
Ganga AR serves as the Co-Founder and Research Strategist of QuantNova, playing a pivotal role in strengthening the analytical integrity and validation discipline of the platform. While system architecture forms the structural backbone of QuantNova, research rigor forms its intellectual core — and Ganga’s contributions are central to that foundation.

Her work focuses extensively on validation methodology, structured documentation practices, and analytical refinement. She ensures that experimental results are not merely generated but critically evaluated. By emphasizing reproducibility and systematic testing, she reinforces the platform’s commitment to academic robustness.

Ganga approaches quantitative modeling from a structured research perspective. She examines assumptions, challenges model biases, and contributes to refining the interpretability of predictive outcomes. Her involvement ensures that QuantNova does not drift toward overfitting or speculative modeling practices, but instead remains anchored in disciplined statistical reasoning.

Beyond validation, Ganga contributes to the long-term research roadmap of the platform. She actively participates in defining experimentation cycles, refining hypothesis testing frameworks, and strengthening the documentation standards that support continuous platform evolution. Her focus on clarity, structure, and analytical transparency enhances the intellectual credibility of QuantNova.

As a co-founder, she embodies the balance between ambition and responsibility. Her perspective reinforces the principle that scalable AI systems must be built with accountability, structured evaluation, and measurable performance at their core.

Through her research-centered contributions, QuantNova evolves not only as a technical system but as a disciplined academic intelligence platform designed to withstand scrutiny, complexity, and future expansion.
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
