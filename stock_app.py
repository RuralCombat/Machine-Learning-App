import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Stock Technical Analysis & Prediction")

st.title("📊 Stock Technical Analysis & Prediction")

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
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc, X_test, y_test, y_pred

def plot_price_with_indicators(df):
    st.subheader("📈 Closing Price with SMA and EMA")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['Close'], label='Close Price')
    ax.plot(df.index, df['SMA'], label='SMA')
    ax.plot(df.index, df['EMA'], label='EMA')
    ax.set_title("Close Price with SMA & EMA")
    ax.legend()
    st.pyplot(fig)

def plot_feature_importance(model, features):
    st.subheader("📊 Feature Importance")
    importance = model.feature_importances_
    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_xlabel("Importance")
    st.pyplot(fig)

def plot_prediction(y_test, y_pred):
    st.subheader("🔍 Actual vs Predicted Trend")
    fig, ax = plt.subplots()
    ax.plot(y_test.values, label='Actual')
    ax.plot(y_pred, label='Predicted')
    ax.set_title("Actual vs Predicted (1 = Price Up, 0 = Price Down)")
    ax.legend()
    st.pyplot(fig)

if ticker:
    st.write(f"Fetching data for: `{ticker}`")
    df = download_data(ticker)

    if not df.empty:
        df = add_technical_indicators(df)

        if df is not None:
            st.write("✅ Sample Data with Technical Indicators")
            st.dataframe(df.tail())

            try:
                model, acc, X_test, y_test, y_pred = train_model(df)
                st.success(f"🎯 Model trained with {acc:.2%} accuracy.")

                plot_price_with_indicators(df)
                plot_feature_importance(model, X_test.columns)
                plot_prediction(y_test, y_pred)

            except ValueError as e:
                st.error(str(e))
