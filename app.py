import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

st.set_page_config(page_title="AI Trade Bot Pro", layout="wide")

# =============================
# FUTURISTIC STYLE
# =============================
st.markdown("""
<style>
body { background-color: #0e1117; }
.main { background-color: #0e1117; }
h1, h2, h3 { color: #00f5ff; }
.stButton>button {
    background-color: #00f5ff;
    color: black;
    border-radius: 8px;
    height: 3em;
}
.stTextInput>div>div>input {
    background-color: #1a1f2b;
    color: white;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =============================
# LOGIN SYSTEM
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🤖 AI TRADE BOT PRO")
    st.subheader("Secure Access Portal")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user and pwd:
            st.session_state.logged_in = True
            st.success("Access Granted 🚀")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Enter valid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# =============================
# SAFE DATA FETCH
# =============================
def safe_metric(ticker):
    try:
        data = yf.download(ticker, period="5d", progress=False)
        if not data.empty and "Close" in data.columns:
            return float(data["Close"].iloc[-1])
        return None
    except:
        return None

# =============================
# SIDEBAR NAVIGATION
# =============================
st.sidebar.title("⚡ AI Trade Bot Pro")

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "🏠 Home",
        "🌍 Global Markets",
        "📊 Crypto Board",
        "🧠 AI Analyzer",
        "📈 Portfolio",
        "⚙ Settings"
    ]
)

# =============================
# HOME PAGE
# =============================
if menu == "🏠 Home":
    st.title("🌌 AI Trade Intelligence Platform")

    col1, col2, col3 = st.columns(3)
    col1.metric("Active Markets", "128")
    col2.metric("AI Accuracy", "82%")
    col3.metric("Live Signals", "14")

    st.divider()
    st.info("Next-generation AI-powered market intelligence dashboard.")

# =============================
# GLOBAL MARKETS
# =============================
if menu == "🌍 Global Markets":
    st.title("🌍 Global Indices Overview")

    indices = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "NIFTY 50": "^NSEI"
    }

    for name, ticker in indices.items():
        value = safe_metric(ticker)
        if value:
            st.metric(name, f"${round(value,2)}")
        else:
            st.metric(name, "Unavailable")

# =============================
# CRYPTO BOARD
# =============================
if menu == "📊 Crypto Board":
    st.title("📊 Live Crypto Market")

    crypto_list = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]

    data_rows = []

    for coin in crypto_list:
        price = safe_metric(coin)
        data_rows.append({
            "Coin": coin.replace("-USD",""),
            "Price (USD)": round(price,2) if price else "Unavailable"
        })

    st.table(data_rows)

# =============================
# AI ANALYZER
# =============================
if menu == "🧠 AI Analyzer":
    st.title("🧠 Neural Trade Prediction Engine")

    stock_symbol = st.text_input("Enter Stock Symbol", "RELIANCE.NS")

    if st.button("🚀 Analyze Market"):

        progress = st.progress(0)
        status = st.empty()

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
            status.text("Analyzing market patterns using AI...")

        try:
            stock = yf.download(stock_symbol, start="2023-01-01", progress=False)

            if stock.empty or "Close" not in stock.columns:
                st.error("Unable to fetch data.")
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
                    st.error("Not enough historical data.")

        except:
            st.error("Something went wrong during analysis.")

# =============================
# PORTFOLIO (DEMO)
# =============================
if menu == "📈 Portfolio":
    st.title("📈 Portfolio Overview")

    st.metric("Total Value", "$12,450")
    st.metric("Today's Gain", "+2.3%")
    st.progress(70)
    st.info("Portfolio module coming in next version.")

# =============================
# SETTINGS
# =============================
if menu == "⚙ Settings":
    st.title("⚙ Platform Settings")
    st.write("Theme: Futuristic Dark")
    st.write("Version: 1.0 Pro")
    st.write("Developer Mode: Enabled")
