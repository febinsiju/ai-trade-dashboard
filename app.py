import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Quant Trading Lab",
    layout="wide",
    page_icon="📊"
)

st.title("📊 AI Quantitative Trading Research Lab")

st.markdown("Institutional-grade stock prediction and strategy validation system.")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Model Settings")

stock_symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
n_estimators = st.sidebar.slider("Model Complexity", 50, 300, 100)
test_size = st.sidebar.slider("Test Data Size (%)", 10, 40, 20)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data(symbol):
    data = yf.download(symbol, period="2y")
    return data

data = load_data(stock_symbol)

if data.empty:
    st.error("Invalid stock symbol.")
    st.stop()

# ==========================================
# FEATURE ENGINEERING
# ==========================================

data["SMA_10"] = data["Close"].rolling(10).mean()
data["SMA_50"] = data["Close"].rolling(50).mean()
data["Return"] = data["Close"].pct_change()
data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)

data = data.dropna()

features = ["SMA_10", "SMA_50", "Return"]
X = data[features]
y = data["Target"]

# ==========================================
# TRAIN MODEL
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, shuffle=False
)

model = RandomForestClassifier(n_estimators=n_estimators)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# ==========================================
# TABS
# ==========================================

tab1, tab2, tab3 = st.tabs(["📈 Price Chart", "🧠 AI Prediction", "📊 Backtest"])

# ==========================================
# TAB 1 – PRICE CHART
# ==========================================

with tab1:
    st.subheader("Stock Price with Moving Averages")

    fig, ax = plt.subplots()
    ax.plot(data["Close"], label="Close Price")
    ax.plot(data["SMA_10"], label="SMA 10")
    ax.plot(data["SMA_50"], label="SMA 50")
    ax.legend()

    st.pyplot(fig)

# ==========================================
# TAB 2 – PREDICTION
# ==========================================

with tab2:
    st.subheader("Next Day Prediction")

    latest_data = X.iloc[-1:].values
    prediction = model.predict(latest_data)[0]
    probability = model.predict_proba(latest_data)[0]

    if prediction == 1:
        st.success("📈 AI Signal: BUY")
        confidence = round(probability[1] * 100, 2)
    else:
        st.error("📉 AI Signal: SELL")
        confidence = round(probability[0] * 100, 2)

    st.metric("Model Accuracy", f"{round(accuracy*100,2)}%")
    st.metric("Confidence Level", f"{confidence}%")

# ==========================================
# TAB 3 – BACKTEST
# ==========================================

with tab3:
    st.subheader("Strategy Backtest")

    data_test = data.iloc[-len(X_test):].copy()
    data_test["Prediction"] = predictions

    data_test["Strategy_Return"] = data_test["Return"] * data_test["Prediction"]
    data_test["Cumulative_Market"] = (1 + data_test["Return"]).cumprod()
    data_test["Cumulative_Strategy"] = (1 + data_test["Strategy_Return"]).cumprod()

    fig2, ax2 = plt.subplots()
    ax2.plot(data_test["Cumulative_Market"], label="Buy & Hold")
    ax2.plot(data_test["Cumulative_Strategy"], label="AI Strategy")
    ax2.legend()

    st.pyplot(fig2)

    strategy_return = round((data_test["Cumulative_Strategy"].iloc[-1] - 1) * 100, 2)
    market_return = round((data_test["Cumulative_Market"].iloc[-1] - 1) * 100, 2)

    st.metric("AI Strategy Return (%)", strategy_return)
    st.metric("Market Return (%)", market_return)

st.markdown("---")
st.markdown("⚠️ Educational research model. Not financial advice.")
