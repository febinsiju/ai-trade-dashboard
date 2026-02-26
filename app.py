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
# IMAGE FUNCTION (NO STRETCH + CIRCULAR)
# =====================================================

def circular_image(image_path, size=180):
    if not os.path.exists(image_path):
        st.warning(f"{image_path} not found")
        return

    img = Image.open(image_path)

    # Crop to square (center crop)
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
        <div style="text-align:center; margin-bottom:20px;">
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
# SIDEBAR
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    ["Home", "AI Engine", "Backtesting Laboratory", "About Us", "Contact", "Follow Us"]
)

# =====================================================
# HOME
# =====================================================

if page == "Home":

    st.title("QuantNova AI Trading Intelligence Platform")

    st.markdown("""
    QuantNova is a structured quantitative research initiative designed to explore
    the intersection of artificial intelligence, financial modeling, and statistical validation.

    The platform is built upon disciplined experimentation, incorporating ensemble learning
    methods and structured feature engineering to convert historical market behavior into
    probabilistic predictive insights.

    Rather than promoting speculation, QuantNova emphasizes data-driven reasoning,
    controlled validation pipelines, and systematic evaluation of predictive confidence.
    """)

    st.info("Developed strictly for academic research and demonstration purposes.")

# =====================================================
# AI ENGINE
# =====================================================

elif page == "AI Engine":

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

elif page == "Backtesting Laboratory":

    st.title("Strategy Backtesting Laboratory")

    symbol = st.text_input("Stock Symbol", "AAPL")

    data = yf.download(symbol, period="2y")

    if data.empty:
        st.error("Invalid stock symbol.")
        st.stop()

    data["Return"] = data["Close"].pct_change()
    data = data.dropna()
    data["Market Growth"] = (1 + data["Return"]).cumprod()

    fig, ax = plt.subplots()
    ax.plot(data["Market Growth"])
    ax.set_title("Buy & Hold Performance")
    st.pyplot(fig)

# =====================================================
# ABOUT US
# =====================================================

elif page == "About Us":

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

    # Founder Section
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
    financial time-series analysis with machine learning algorithms such as
    Random Forest classifiers, he aimed to construct a system capable of evolving
    through iterative exposure to market data.

    Beyond coding implementation, Febin directed the architectural blueprint of the
    platform — defining modular components, ensuring data preprocessing integrity,
    and aligning the research objectives with academic rigor.

    His long-term vision for QuantNova involves expanding toward adaptive learning
    frameworks, incorporating additional technical indicators, and exploring
    probabilistic confidence calibration methods to enhance decision-support reliability.
    """)

    st.markdown("---")

    # Co-Founder Section
    circular_image("ganga_image.jpg", 180)
    st.markdown("<h3 style='text-align:center;'>Ganga AR</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-weight:600;'>Co-Founder & Research Strategist</p>", unsafe_allow_html=True)

    st.markdown("""
    Ganga AR serves as Co-Founder and Research Strategist for QuantNova. Her
    contributions focused on analytical validation, structured evaluation of
    model outputs, and refinement of documentation standards to ensure clarity
    and academic integrity.

    She played a critical role in assessing model performance metrics,
    interpreting predictive confidence levels, and ensuring that the platform
    maintained a disciplined, research-oriented perspective rather than
    speculative positioning.

    Ganga contributed significantly to strengthening the theoretical
    foundations of the initiative, aligning implementation with
    established machine learning principles and structured testing methodology.

    Her involvement ensured that QuantNova remained not only a functional
    software system but also a well-documented research framework capable
    of academic presentation and further development.
    """)

# =====================================================
# CONTACT
# =====================================================

elif page == "Contact":

    st.title("Contact")

    st.write("Email: quantnova.ai@gmail.com")
    st.write("Institution: TocH Institute Of Science And Technology")
    st.write("Location: Ernakulam, Kerala")

# =====================================================
# FOLLOW
# =====================================================

elif page == "Follow Us":

    st.title("Follow Us")

    st.write("LinkedIn")
    st.write("Instagram")
    st.write("Twitter")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown(f"© {datetime.datetime.now().year} QuantNova AI Research Initiative")
