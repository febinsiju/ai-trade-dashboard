# -*- coding: utf-8 -*-
import streamlit as st
...
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
import streamlit.components.v1 as components
import streamlit as st
import sqlite3
import hashlib
# -*- coding: utf-8 -*-

import streamlit as st
import streamlit.components.v1 as components
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
import sqlite3
import hashlib

# =====================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# =====================================================

st.set_page_config(layout="wide")

# =====================================================
# ANIMATED FUTURISTIC BACKGROUND
# =====================================================

st.markdown("""
<style>

/* ================================
   MAIN BACKGROUND ANIMATION
================================ */

.stApp {
    background: linear-gradient(-45deg, #0f172a, #111827, #0b1220, #0e1a2b);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
    overflow-x: hidden;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ================================
   FLOATING GLOW ORBS
================================ */

.orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(90px);
    opacity: 0.6;
    z-index: 0;
    animation: floatMove 20s infinite alternate ease-in-out;
}

.orb1 {
    width: 400px;
    height: 400px;
    background: rgba(0, 200, 255, 0.4);
    top: 10%;
    left: 5%;
}

.orb2 {
    width: 350px;
    height: 350px;
    background: rgba(0, 255, 163, 0.3);
    bottom: 10%;
    right: 10%;
    animation-delay: 4s;
}

.orb3 {
    width: 300px;
    height: 300px;
    background: rgba(138, 43, 226, 0.35);
    top: 50%;
    left: 40%;
    animation-delay: 8s;
}

@keyframes floatMove {
    0% { transform: translateY(0px) translateX(0px) scale(1); }
    50% { transform: translateY(-40px) translateX(30px) scale(1.1); }
    100% { transform: translateY(30px) translateX(-20px) scale(1); }
}

/* ================================
   MAKE CONTENT ABOVE BACKGROUND
================================ */

.block-container {
    position: relative;
    z-index: 2;
}

/* ================================
   SIDEBAR
================================ */

section[data-testid="stSidebar"] {
    background: rgba(10,15,28,0.85);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* ================================
   BUTTONS
================================ */

.stButton > button {
    background: linear-gradient(90deg, #00C8FF, #00FFA3);
    color: black;
    border-radius: 10px;
    font-weight: 600;
    border: none;
    padding: 10px 20px;
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 20px rgba(0,255,163,0.6);
}

/* ================================
   HERO TITLE
================================ */

.main-title {
    font-size: 52px;
    font-weight: 800;
    text-align: center;
    color: white;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: rgba(255,255,255,0.7);
    margin-bottom: 40px;
}

</style>

<!-- Floating Orbs -->
<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>

""", unsafe_allow_html=True)

# =====================================================
# CLEAN BIG LOGO DISPLAY
# =====================================================

def load_logo(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = load_logo("quantnova_logo.png")

st.markdown(f"""
<div style="text-align:center; margin-top:60px; margin-bottom:20px;">
    <img src="data:image/png;base64,{logo_base64}"
    style="
        width:260px;
        max-width:85%;
        filter: drop-shadow(0 0 40px rgba(0,200,255,0.5));
    ">
</div>

<div class="main-title">QuantNova</div>
<div class="subtitle">
AI-Powered Quantitative Intelligence Platform
</div>
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

pages = ["Home", "AI Intelligence Engine", "Strategy Lab", "Market Dashboard", "About"]

if st.session_state.page in pages:
    idx = pages.index(st.session_state.page)
else:
    idx = 0

selected = st.sidebar.radio("Navigate", pages, index=idx)

if selected != st.session_state.page:
    st.session_state.page = selected
    st.rerun()

if "username" in st.session_state:
    st.sidebar.write(f"Logged in as: {st.session_state.username}")

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


        # =====================================================
    # FULL PLATFORM OVERVIEW
    # =====================================================

    st.header("AI Intelligence Engine")

    st.markdown("""
The AI Intelligence Engine is the computational core of QuantNova. It is designed to process complex, multi-dimensional financial datasets and transform them into probability-driven insight frameworks.

Rather than relying on static indicators or traditional technical analysis, the engine applies adaptive machine learning models, volatility clustering diagnostics, cross-asset correlation mapping, and structural break detection algorithms. 
This allows the system to continuously recalibrate as new information enters the market environment.

By integrating predictive modeling with real-time statistical validation, the engine reduces signal noise and enhances informational precision. 
It identifies momentum transitions, liquidity imbalances, and regime shifts before they become widely recognized by the market.
""")

    st.markdown("---")

    st.header("Strategy Lab")

    st.markdown("""
The Strategy Lab serves as the experimental research and validation layer of QuantNova. It is a controlled quantitative environment where trading hypotheses evolve into rigorously tested systematic frameworks.

Inside the lab, strategies undergo parameter optimization, walk-forward testing, Monte Carlo simulations, and drawdown sensitivity analysis. 
Each model is evaluated across multiple volatility regimes to ensure structural robustness rather than curve-fitted performance.

This architecture enables disciplined experimentation while maintaining mathematical integrity transforming theoretical ideas into deployable quantitative systems.
""")

    st.markdown("---")

    st.header("Market Dashboard")

    st.markdown("""
The Market Dashboard provides structured situational awareness across macro and micro timeframes. 
It consolidates volatility metrics, momentum shifts, sector rotation dynamics, and liquidity flow structures into a unified analytical interface.

Designed for clarity rather than clutter, the dashboard surfaces only materially relevant signals — reducing cognitive overload while enhancing decision accuracy.

Integrated risk diagnostics ensure that exposure levels are evaluated relative to prevailing market conditions, allowing users to maintain strategic alignment in dynamic environments.
""")

    st.markdown("---")

    if st.button("About Us"):
        st.session_state.page = "About"
        st.rerun()

    st.markdown("---")

    st.header("Contact Us")
    st.write("quantnova.ai.com")

    st.header("Follow Us On")

    st.markdown("""
    <style>
    .insta-link {
        display: block;
        font-size: 15px;
        margin: 6px 0;
        color: white;
        text-decoration: none;
        transition: 0.3s ease;
        font-weight: 400;
    }

    .insta-link:hover {
        color: white;
        text-shadow: 0 0 8px rgba(0,255,163,0.8);
        transform: translateX(4px);
    }
    </style>

    <a href="https://www.instagram.com/f_eb_in_" target="_blank" class="insta-link">f_eb_in_</a>
    <a href="https://www.instagram.com/_gan.ga___?utm_source=ig_web_button_share_sheet&igsh=ZDNlZDc0MzIxNw==" target="_blank" class="insta-link">gan.ga</a>
    <a href="https://www.instagram.com/fiza.farshad" target="_blank" class="insta-link">fiza.farshad</a>
    <a href="https://www.instagram.com/its_g4nia" target="_blank" class="insta-link">its_g4nia</a>
    """, unsafe_allow_html=True)
# =====================================================
# AI ENGINE
# =====================================================

elif st.session_state.page == "AI Intelligence Engine":

    st.title("AI Intelligence Engine — Model Leaderboard")

    import yfinance as yf
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import plotly.graph_objects as go

    symbol = st.text_input("Enter Stock Symbol", "AAPL")

    if st.button("Run QuantNova AI Engine"):

        with st.spinner("Initializing QuantNova Neural Core..."):

            # -----------------------------
            # Download Data (Cloud Safe)
            # -----------------------------
            try:
                data = yf.download(symbol, period="5y", auto_adjust=True)
            except Exception:
                st.error("Error fetching market data.")
                st.stop()

            if data.empty:
                st.error("No data found. Please enter a valid stock symbol.")
                st.stop()

            # Fix possible multi-index columns (Streamlit Cloud issue)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if "Close" not in data.columns:
                st.error("Close price not found in dataset.")
                st.stop()

            # -----------------------------
            # Feature Engineering
            # -----------------------------
            data["Return"] = data["Close"].pct_change()
            data["Target"] = np.where(data["Return"] > 0, 1, 0)

            data["MA10"] = data["Close"].rolling(10).mean()
            data["MA50"] = data["Close"].rolling(50).mean()
            data["Volatility"] = data["Return"].rolling(10).std()

            data = data.dropna()

            if len(data) < 30:
                st.error("Insufficient data after preprocessing.")
                st.stop()

            # -----------------------------
            # Train/Test Split
            # -----------------------------
            X = data[["MA10", "MA50", "Volatility"]]
            y = data["Target"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, shuffle=False
            )

            # -----------------------------
            # Model Definitions
            # -----------------------------
            models = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Support Vector Machine": SVC(probability=True)
            }

            leaderboard = []
            best_model = None
            best_score = 0
            best_name = ""

            # -----------------------------
            # Train & Evaluate
            # -----------------------------
            for name, model in models.items():

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)

                leaderboard.append((name, accuracy))

                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model
                    best_name = name

            leaderboard_df = pd.DataFrame(
                leaderboard,
                columns=["Model", "Accuracy"]
            ).sort_values("Accuracy", ascending=False)

            st.subheader("AI Model Leaderboard")
            st.dataframe(leaderboard_df)

            st.success(f"Best Performing Model: {best_name}")

            # -----------------------------
            # Latest AI Signal
            # -----------------------------
            probabilities = best_model.predict_proba(X_test)[:, 1]

            st.subheader("Latest AI Signal")

            col1, col2, col3 = st.columns(3)
            col1.metric("Best Model", best_name)
            col2.metric("Accuracy", f"{best_score*100:.2f}%")
            col3.metric(
                "Signal",
                "BUY" if probabilities[-1] > 0.5 else "SELL"
            )

            # -----------------------------
            # Strategy Backtest
            # -----------------------------
            data_test = data.iloc[-len(probabilities):].copy()
            data_test["AI_Strategy"] = (
                data_test["Return"] * (probabilities > 0.5)
            )

            cumulative_market = (1 + data_test["Return"]).cumprod()
            cumulative_ai = (1 + data_test["AI_Strategy"]).cumprod()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=cumulative_market,
                name="Market"
            ))
            fig.add_trace(go.Scatter(
                y=cumulative_ai,
                name="AI Strategy"
            ))

            fig.update_layout(title="Best Model Strategy vs Market")
            st.plotly_chart(fig, use_container_width=True)

            # -----------------------------
            # Risk Metrics
            # -----------------------------
            sharpe = (
                data_test["AI_Strategy"].mean() /
                data_test["AI_Strategy"].std()
            ) * np.sqrt(252)

            volatility = (
                data_test["AI_Strategy"].std() * np.sqrt(252)
            )

            drawdown = (
                cumulative_ai /
                cumulative_ai.cummax() - 1
            ).min()

            st.subheader("Risk Metrics")

            r1, r2, r3 = st.columns(3)
            r1.metric("Sharpe Ratio", f"{sharpe:.2f}")
            r2.metric("Volatility", f"{volatility:.2f}")
            r3.metric("Max Drawdown", f"{drawdown:.2%}")

# =====================================================
# STRATEGY LAB
# =====================================================

elif st.session_state.page == "Strategy Lab":

    st.title("Strategy Lab — Multi Strategy Engine")

    import yfinance as yf
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    symbol = st.text_input("Enter Stock Symbol", "AAPL", key="multi_strategy_symbol")

    if st.button("Run Quant Strategy Engine"):

        with st.spinner("Running Multi-Strategy Backtest..."):

            data = yf.download(symbol, period="2y")
            data["Return"] = data["Close"].pct_change()

            # -------------------
            # Strategy 1: Moving Average
            # -------------------
            data["MA10"] = data["Close"].rolling(10).mean()
            data["MA50"] = data["Close"].rolling(50).mean()
            data["MA_Signal"] = np.where(data["MA10"] > data["MA50"], 1, 0)
            data["MA_Strategy"] = data["Return"] * data["MA_Signal"]

            # -------------------
            # Strategy 2: RSI
            # -------------------
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            data["RSI"] = 100 - (100 / (1 + rs))
            data["RSI_Signal"] = np.where(data["RSI"] < 30, 1, 0)
            data["RSI_Strategy"] = data["Return"] * data["RSI_Signal"]

            # -------------------
            # Strategy 3: Buy & Hold
            # -------------------
            data["BuyHold"] = data["Return"]

            data = data.dropna()

            cumulative_market = (1 + data["BuyHold"]).cumprod()
            cumulative_ma = (1 + data["MA_Strategy"]).cumprod()
            cumulative_rsi = (1 + data["RSI_Strategy"]).cumprod()

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=cumulative_market, name="Market"))
            fig.add_trace(go.Scatter(y=cumulative_ma, name="Moving Average"))
            fig.add_trace(go.Scatter(y=cumulative_rsi, name="RSI"))

            fig.update_layout(title="Multi-Strategy Performance Comparison")
            st.plotly_chart(fig, use_container_width=True)

            # Performance Metrics Table
            def metrics(series):
                sharpe = (series.mean() / series.std()) * np.sqrt(252)
                volatility = series.std() * np.sqrt(252)
                total_return = (1 + series).cumprod().iloc[-1] - 1
                return sharpe, volatility, total_return

            ma_metrics = metrics(data["MA_Strategy"])
            rsi_metrics = metrics(data["RSI_Strategy"])
            bh_metrics = metrics(data["BuyHold"])

            performance_table = pd.DataFrame({
                "Strategy": ["Moving Average", "RSI", "Buy & Hold"],
                "Sharpe Ratio": [ma_metrics[0], rsi_metrics[0], bh_metrics[0]],
                "Volatility": [ma_metrics[1], rsi_metrics[1], bh_metrics[1]],
                "Total Return": [ma_metrics[2], rsi_metrics[2], bh_metrics[2]]
            })

            st.subheader("Strategy Performance Leaderboard")
            st.dataframe(performance_table.sort_values("Sharpe Ratio", ascending=False))

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

# =====================================================
# ABOUT PAGE
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
    # CSS STYLING
    # =============================
    st.markdown("""
    <style>

    .team-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(18px);
        border-radius: 25px;
        padding: 40px 30px;
        margin: 60px auto;
        max-width: 850px;
        transition: all 0.4s ease;
    }

    .team-card:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateY(-10px);
        box-shadow: 0 0 40px rgba(0, 200, 255, 0.25);
    }

    .profile-container {
        position: relative;
        width: 220px;
        height: 220px;
        margin: 0 auto 30px auto;
        animation: float 6s ease-in-out infinite;
    }

    .profile-image {
        width: 100%;
        height: 100%;
        border-radius: 50%;
        object-fit: cover;
        object-position: center;
        display: block;
        transition: all 0.4s ease;
        border: 3px solid rgba(0, 200, 255, 0.3);
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
        padding: 20px;
        font-size: 14px;
        transition: all 0.4s ease;
    }

    .profile-container:hover .profile-overlay {
        opacity: 1;
    }

    .profile-container:hover .profile-image {
        transform: scale(1.07);
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
        100% { transform: translateY(0px); }
    }

    h3 {
        text-align: center;
        font-size: 22px;
        margin-bottom: 5px;
    }

    p.role-title {
        text-align: center;
        font-weight: 600;
        letter-spacing: 1px;
        color: #00C8FF;
        margin-bottom: 25px;
    }

    </style>
    """, unsafe_allow_html=True)

    # =============================
    # IMAGE FUNCTION
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
    <div class="team-card">

    <div class="profile-container">
        <img src="data:image/png;base64,{founder_img}" class="profile-image">
        <div class="profile-overlay">
            Founder & Lead Architect of QuantNova.<br><br>
            Designed AI architecture, predictive systems,
            and long-term intelligence infrastructure roadmap.
        </div>
    </div>

    <h3>Febin Siju</h3>
    <p class="role-title">Founder & Lead Architect</p>

    </div>
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
    <div class="team-card">

    <div class="profile-container">
        <img src="data:image/png;base64,{cofounder_img}" class="profile-image">
        <div class="profile-overlay">
            Co-Founder & Research Strategist of QuantNova.<br><br>
            Leads validation methodology, structured experimentation,
            and analytical integrity across predictive systems.
        </div>
    </div>

    <h3>Ganga AR</h3>
    <p class="role-title">Co-Founder & Research Strategist</p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Ganga AR strengthens QuantNova’s research discipline through structured validation frameworks, reproducible experimentation processes, and rigorous analytical documentation. Her focus ensures the platform maintains academic integrity while evolving toward scalable AI intelligence infrastructure.
    """)

    st.markdown("---")

    # =============================
    # CHIEF TECHNOLOGY ENGINEER
    # =============================
    fiza_img = get_base64_image("fiza_image.jpg")

    st.markdown(f"""
    <div class="team-card">

    <div class="profile-container">
        <img src="data:image/png;base64,{fiza_img}" class="profile-image">
        <div class="profile-overlay">
            Chief Technology Engineer.<br><br>
            Leads system optimization, backend architecture,
            and infrastructure scalability.
        </div>
    </div>

    <h3>Fiza KF</h3>
    <p class="role-title">Chief Technology Engineer</p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Fiza plays a critical role in transforming QuantNova’s conceptual AI frameworks into stable, production-ready systems. As Chief Technology Engineer, she is responsible for backend system optimization, database architecture structuring, and performance-level engineering decisions that ensure computational efficiency.

    Her contributions extend into building modular pipelines that allow scalable model experimentation without compromising system stability. She focuses on reducing latency in model execution, improving data ingestion workflows, and maintaining secure authentication mechanisms within the platform.

    By combining strong engineering discipline with analytical awareness, she ensures QuantNova operates not just as a research prototype, but as a scalable SaaS intelligence infrastructure capable of handling increasing computational complexity and expanding feature layers.
    """)

    st.markdown("---")

    # =============================
    # HEAD OF DATA SCIENCE
    # =============================
    gania_img = get_base64_image("gania_image.jpeg")

    st.markdown(f"""
    <div class="team-card">

    <div class="profile-container">
        <img src="data:image/png;base64,{gania_img}" class="profile-image">
        <div class="profile-overlay">
            Head of Data Science.<br><br>
            Oversees model validation, feature engineering,
            and statistical research frameworks.
        </div>
    </div>

    <h3>Gania Gibu</h3>
    <p class="role-title">Head of Data Science</p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Gania leads QuantNova’s data science initiatives with a focus on predictive accuracy, statistical robustness, and reproducible experimentation. As Head of Data Science, she oversees feature engineering strategies, volatility modeling techniques, and classification framework validation.

    Her expertise lies in transforming raw market data into structured analytical features that enhance model performance while minimizing overfitting risks. She ensures that every predictive structure undergoes rigorous validation processes, including cross-validation testing, sensitivity analysis, and regime-specific performance evaluation.

    By integrating disciplined research methodology with applied machine learning, she strengthens the probabilistic foundations of QuantNova. Her role ensures that the platform’s AI outputs are grounded in measurable statistical evidence rather than speculative interpretation.
    """)
# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown(f"© {datetime.datetime.now().year} QuantNova | SaaS Research Build v1.0")
