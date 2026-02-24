import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

st.set_page_config(page_title="AI Trade Bot Pro", layout="wide")

# =============================
# FUTURISTIC STYLING
# =============================
st.markdown("""
<style>
body { background-color: #0e1117; }
.main { background-color: #0e1117; }
h1, h2, h3 { color: #00f5ff; }
.stButton>button {
    background-color: #00f5ff;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stTextInput>div>div>input {
    background-color: #1a1f2b;
    color: white;
}
.metric-card {
    background-color: #1a1f2b;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 0 15px #00f5ff;
}
</style>
""", unsafe_allow_html=True)

# =============================
# LOGIN SYSTEM
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🤖 AI TRADE BOT PRO")
    st.subheader("Secure Access Portal")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password:
            st.session_state.logged_in = True
            st.success("Access Granted 🚀")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Invalid Credentials")

if not st.session_state.logged_in:
    login_page()
    st.stop()

# =============================
# SIDEBAR
# =============================
st.sidebar.title("⚡ Navigation")
page = st.sidebar.radio("Select Module", ["Dashboard", "AI Analyzer"])

st.title("🌌 AI Global Trade Intelligence")

# =============================
# SAFE DATA FETCH FUNCTION
# =============================
def safe_metric(ticker):
    try:
        data = yf.download(ticker, period="5d")
        if not data.empty and "Close" in data.columns:
            return float(data["Close"].iloc[-1])
        return None
    except:
        return None

# =============================
# DASHBOARD
# =============================
if page == "Dashboard":

    st.subheader("📊 Live Global Market")

    col1, col2, col3 = st.columns(3)

    btc = safe_metric("BTC-USD")
    eth = safe_metric("ETH-USD")
    gold = safe_metric("GC=F")

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if btc is not None:
            st.metric("Bitcoin (BTC)", f"${round(btc,2)}")
        else:
            st.metric("Bitcoin (BTC)", "Unavailable")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if eth is not None:
            st.metric("Ethereum (ETH)", f"${round(eth,2)}")
        else:
            st.metric("Ethereum (ETH)", "Unavailable")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if gold is not None:
            st.metric("Gold", f"${round(gold,2)}")
        else:
            st.metric("Gold", "Unavailable")
        st.markdown('</div>', unsafe_allow_html=True)

# =============================
# AI ANALYZER
# =============================
if page == "AI Analyzer":

    st.subheader("🧠 Neural Trade Prediction Engine")

    stock_symbol = st.text_input("Enter Stock Symbol (Example: RELIANCE.NS)", "RELIANCE.NS")

    if st.button("🚀 Analyze Market"):

        progress = st.progress(0)
        status = st.empty()

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
            status.text("Analyzing market patterns using AI...")

        try:
            stock = yf.download(stock_symbol, start="2023-01-01")

            if stock.empty or "Close" not in stock.columns:
                st.error("Unable to fetch stock data.")
            else:
                stock["MA10"] = stock["Close"].rolling(10).mean()
                stock["MA50"] = stock["Close"].rolling(50).mean()
                stock["Target"] = (stock["Close"].shift(-1) > stock["Close"]).astype(int)
                stock = stock.dropna()

                if len(stock) > 50:
                    X = stock[["MA10", "MA50"]]
                    y = stock["Target"]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False
                    )

                    model = RandomForestClassifier(n_estimators=100)
                    model.fit(X_train, y_train)

                    latest = stock[["MA10", "MA50"]].iloc[-1:]
                    prediction = model.predict(latest)
                    probability = model.predict_proba(latest)
                    confidence = round(np.max(probability) * 100, 2)

                    st.success("AI Analysis Complete ✅")

                    if prediction[0] == 1:
                        st.success(f"📈 BUY Signal (Confidence: {confidence}%)")
                    else:
                        st.error(f"📉 SELL Signal (Confidence: {confidence}%)")

                    st.line_chart(stock["Close"])
                else:
                    st.error("Not enough historical data for analysis.")

        except Exception as e:
            st.error("Something went wrong during analysis.")
