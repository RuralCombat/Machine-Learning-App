import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="📊 Stock Technical Analysis & Prediction")

st.title("📊 Stock Technical Analysis & Prediction")

# Input
ticker = st.text_input("Enter Stock Ticker (e.g., WIPRO.NS):")

@st.cache_data
def download_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        return df
    except Exception as e:
        st.error(f"Download error: {e}")
        return pd.DataFrame()

def add_technical_indicators(df):
    try:
        close = df['Close'].squeeze()
        volume = df['Volume'].squeeze()

        df['SMA'] = ta.trend.sma_indicator(close)
        df['EMA'] = ta.trend.ema_indicator(close)
        df['RSI'] = ta.momentum.rsi(close)
        df['MACD'] = ta.trend.macd_diff(close)
        df['BB_H'] = ta.volatility.bollinger_hband(close)
        df['BB_L'] = ta.volatility.bollinger_lband(close)
        df['Volume_SMA'] = ta.trend.sma_indicator(volume)

        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        df.dropna(inplace=True)

        return df
    except Exception as e:
        st.error(f"Error computing indicators: {e}")
        return None

def train_model(df):
    features = ['SMA', 'EMA', 'RSI', 'MACD', 'BB_H', 'BB_L', 'Volume_SMA']
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for model training: {missing}")

    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    return model, acc

if ticker:
    st.write(f"Fetching data for: `{ticker}`")
    df = download_data(ticker)

    if not df.empty:
        df = add_technical_indicators(df)

        if df is not None:
            st.write("Sample data with indicators:")
            st.dataframe(df.tail())

            try:
                model, acc = train_model(df)
                st.success(f"✅ Model trained successfully!\n📈 Accuracy: {acc:.2%}")
            except ValueError as e:
                st.error(str(e))
