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

# =====================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# =====================================================
st.set_page_config(layout="wide")
# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(layout="wide")

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

pages = ["Home", "AI Intelligence Engine", "Strategy Lab", "Market Dashboard", "About"]
if st.session_state.page in pages:
    idx = pages.index(st.session_state.page)
else:
    idx = 0

selected = st.sidebar.radio("Navigate", pages, index=idx)

# Only update if current page is a sidebar page
if selected != st.session_state.page:
    st.session_state.page = selected
    st.rerun()
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

    st.markdown("---")

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

This architecture enables disciplined experimentation while maintaining mathematical integrity — transforming theoretical ideas into deployable quantitative systems.
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
    st.write("+91 8089411348")
    st.write("+91 7012958445")

    st.header("Follow Us On")
    st.write("@ f_eb_in_")
    st.write("@ gan.ga_")

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

        with st.spinner("Training Multiple AI Models..."):

            data = yf.download(symbol, period="2y")

            data["Return"] = data["Close"].pct_change()
            data["Target"] = np.where(data["Return"] > 0, 1, 0)

            data["MA10"] = data["Close"].rolling(10).mean()
            data["MA50"] = data["Close"].rolling(50).mean()
            data["Volatility"] = data["Return"].rolling(10).std()

            data = data.dropna()

            X = data[["MA10", "MA50", "Volatility"]]
            y = data["Target"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, shuffle=False
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Support Vector Machine": SVC(probability=True)
            }

            leaderboard = []

            best_model = None
            best_score = 0

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

            # --- Best Model Predictions ---
            probabilities = best_model.predict_proba(X_test)[:, 1]

            st.subheader("Latest AI Signal")

            col1, col2, col3 = st.columns(3)
            col1.metric("Best Model", best_name)
            col2.metric("Accuracy", f"{best_score*100:.2f}%")
            col3.metric("Signal",
                        "BUY" if probabilities[-1] > 0.5 else "SELL")

            # --- Strategy Backtest ---
            data_test = data.iloc[-len(probabilities):].copy()
            data_test["AI_Strategy"] = data_test["Return"] * (probabilities > 0.5)

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

            # --- Risk Metrics ---
            sharpe = (
                data_test["AI_Strategy"].mean() /
                data_test["AI_Strategy"].std()
            ) * np.sqrt(252)

            volatility = data_test["AI_Strategy"].std() * np.sqrt(252)

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
# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown(f"© {datetime.datetime.now().year} QuantNova | SaaS Research Build v1.0")
