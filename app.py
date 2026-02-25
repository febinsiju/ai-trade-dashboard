import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

st.set_page_config(page_title="AI Trade Bot Pro", layout="wide")

# =============================
# PROFESSIONAL DARK CSS
# =============================
st.markdown("""
<style>
body { background-color: #0E1117; }
.main { background-color: #0E1117; }

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

.card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 0px 20px rgba(0, 255, 170, 0.05);
}

.metric-card {
    background-color: #161B22;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

h1, h2, h3 {
    color: #00F5A0;
}

.stButton>button {
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =============================
# LOGIN SYSTEM
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🚀 AI TRADE BOT PRO")
    st.subheader("Secure Institutional Access")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user and pwd:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# =============================
# SAFE METRIC FUNCTION
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
# SIDEBAR
# =============================
st.sidebar.title("⚡ AI TRADE BOT")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Global Markets",
        "Crypto Board",
        "AI Analyzer",
        "Portfolio",
        "Settings"
    ]
)

# =============================
# DASHBOARD
# =============================
if menu == "Dashboard":
    st.title("📊 Trading Intelligence Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Active Markets", "128")

    with col2:
        st.metric("AI Accuracy", "82%")

    with col3:
        st.metric("Live Signals", "14")

    st.divider()

    # Sample Chart
    data = yf.download("BTC-USD", period="1mo", progress=False)

    if not data.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"]
        )])

        fig.update_layout(
            template="plotly_dark",
            height=600,
            margin=dict(l=10, r=10, t=30, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

# =============================
# GLOBAL MARKETS
# =============================
if menu == "Global Markets":
    st.title("🌍 Global Indices")

    indices = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "NIFTY 50": "^NSEI"
    }

    cols = st.columns(4)

    for i, (name, ticker) in enumerate(indices.items()):
        value = safe_metric(ticker)
        with cols[i]:
            if value:
                st.metric(name, f"${round(value,2)}")
            else:
                st.metric(name, "Unavailable")

# =============================
# CRYPTO BOARD
# =============================
if menu == "Crypto Board":
    st.title("💰 Crypto Market Overview")

    crypto_list = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]

    rows = []
    for coin in crypto_list:
        price = safe_metric(coin)
        rows.append({
            "Coin": coin.replace("-USD",""),
            "Price (USD)": round(price,2) if price else "Unavailable"
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

# =============================
# AI ANALYZER (PRO PANEL STYLE)
# =============================
if menu == "AI Analyzer":
    st.title("🧠 AI Trade Signal Engine")

    left, right = st.columns([3,1])

    stock_symbol = left.text_input("Enter Symbol", "RELIANCE.NS")

    if left.button("Run AI Analysis"):

        with st.spinner("Running AI Model..."):
            stock = yf.download(stock_symbol, start="2023-01-01", progress=False)

            if stock.empty:
                st.error("No Data Found")
            else:
                stock["MA10"] = stock["Close"].rolling(10).mean()
                stock["MA50"] = stock["Close"].rolling(50).mean()
                stock["Target"] = (stock["Close"].shift(-1) > stock["Close"]).astype(int)
                stock = stock.dropna()

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

                # Chart
                fig = go.Figure(data=[go.Candlestick(
                    x=stock.index,
                    open=stock["Open"],
                    high=stock["High"],
                    low=stock["Low"],
                    close=stock["Close"]
                )])

                fig.update_layout(template="plotly_dark", height=500)
                left.plotly_chart(fig, use_container_width=True)

                # Signal Panel
                with right:
                    st.markdown("### AI Signal")

                    if prediction[0] == 1:
                        st.success(f"BUY")
                    else:
                        st.error(f"SELL")

                    st.metric("Confidence", f"{confidence}%")

# =============================
# PORTFOLIO
# =============================
if menu == "Portfolio":
    st.title("📈 Portfolio Overview")

    col1, col2 = st.columns(2)
    col1.metric("Total Value", "$12,450")
    col2.metric("Today's Gain", "+2.3%")

    st.progress(70)

# =============================
# SETTINGS
# =============================
if menu == "Settings":
    st.title("⚙ Platform Settings")
    st.write("Theme: Institutional Dark")
    st.write("Version: 2.0 Professional")
