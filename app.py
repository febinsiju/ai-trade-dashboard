import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="QuantNova AI Trading Intelligence",
    layout="wide"
)

# =====================================================
# SESSION STATE NAVIGATION
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
    font-size: 28px;
    font-weight: 700;
    margin-top: 20px;
}

.profile-role {
    font-size: 18px;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 20px;
}

.profile-text {
    font-size: 18px;
    line-height: 1.8;
    text-align: justify;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HOME PAGE
# =====================================================

if st.session_state.page == "Home":

    st.markdown('<div class="big-title">QuantNova AI Trading Intelligence Platform</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="large-paragraph">
    QuantNova is a quantitative research platform developed to explore the integration
    of artificial intelligence with financial market analytics. Modern financial markets
    operate within highly dynamic environments influenced by institutional capital flows,
    algorithmic execution systems, and global economic variables. Understanding such
    complexity requires structured analytical tools capable of processing large volumes
    of data and extracting meaningful patterns.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="large-paragraph">
    This platform applies machine learning methodologies to historical market data in
    order to construct predictive intelligence models. Rather than relying on intuition
    or speculation, QuantNova emphasizes statistical reasoning, disciplined validation,
    and systematic feature engineering to evaluate directional probabilities.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Intelligent Learning Framework</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="large-paragraph">
    The system follows a structured pipeline beginning with real-time data acquisition,
    followed by transformation into engineered features such as moving averages and
    return differentials. These features are processed through ensemble machine learning
    algorithms that detect recurring structural behavior within historical price movements.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="large-paragraph">
    Predictions are generated with probabilistic confidence measures, allowing users to
    interpret model outputs within a risk-aware framework. The objective is not absolute
    certainty, but structured decision support grounded in historical evidence.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Backtesting and Validation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="large-paragraph">
    QuantNova incorporates comparative backtesting to evaluate the AI-driven strategy
    against traditional buy-and-hold approaches. By analyzing cumulative growth and
    signal consistency, the system demonstrates the importance of disciplined model
    evaluation before practical deployment.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("This platform is developed strictly for academic research and demonstration purposes.")

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

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

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
    data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
    data = data.dropna()

    X = data[["Return"]]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    test = data.iloc[-len(X_test):].copy()
    test["Strategy"] = preds * test["Return"]

    test["Market"] = (1 + test["Return"]).cumprod()
    test["AI"] = (1 + test["Strategy"]).cumprod()

    fig, ax = plt.subplots()
    ax.plot(test["Market"], label="Buy and Hold")
    ax.plot(test["AI"], label="AI Strategy")
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
    artificial intelligence applications in financial market analytics.
    """)

    # Founder
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    st.image("founder_image.jpg", use_container_width=True)
    st.markdown('<div class="profile-name">[Your Name]</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-role">Founder & Lead Architect</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="profile-text">
    The Founder conceptualized and architected the QuantNova platform, designing
    the machine learning framework, data engineering structure, and predictive
    validation methodology. The long-term vision is to expand this initiative
    into a research-oriented AI intelligence system.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Co-Founder
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    st.image("ganga_image.jpg", use_container_width=True)
    st.markdown('<div class="profile-name">Ganga AR</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-role">Co-Founder & Research Strategist</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="profile-text">
    Ganga AR contributed to system validation, model evaluation discussions,
    and strengthening research documentation to ensure academic rigor.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Team Members
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    st.markdown('<div class="profile-name">Fiza KF</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-role">Core Development Member</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-text">Contributed to research support and documentation refinement.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    st.markdown('<div class="profile-name">Gania Gibu</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-role">Conceptual & Interface Support</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-text">Contributed to conceptual clarity and interface refinement.</div>', unsafe_allow_html=True)
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
    st.write("LinkedIn: linkedin.com")
    st.write("Instagram: instagram.com")
    st.write("Twitter: twitter.com")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown(f"© {datetime.datetime.now().year} QuantNova AI Research Initiative")
