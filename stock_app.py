import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import numpy as np

st.set_page_config(page_title="Stock Predictor", layout="centered")

def load_data(ticker):
    st.info(f"Fetching data for: {ticker}")
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)
    if df.empty:
        st.error("Download failed or returned empty data.")
        return None
    return df

def add_technical_indicators(df):
    try:
        df['SMA'] = ta.trend.sma_indicator(df['Close'])
        df['EMA'] = ta.trend.ema_indicator(df['Close'])
        df['RSI'] = ta.momentum.rsi(df['Close'])
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['BB_H'] = ta.volatility.bollinger_hband(df['Close'])
        df['BB_L'] = ta.volatility.bollinger_lband(df['Close'])
        df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'])
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    except Exception as e:
        st.error(f"Error computing indicators: {e}")
        return None

    df.dropna(inplace=True)
    return df

def main():
    st.title("📊 Stock Technical Analysis & Prediction")

    ticker = st.text_input("Enter Stock Ticker (e.g., WIPRO.NS):", "WIPRO.NS")
    
    if st.button("Analyze"):
        df = load_data(ticker)
        if df is not None:
            df = add_technical_indicators(df)
            if df is not None:
                st.success("Indicators computed successfully!")
                st.subheader("Preview of Processed Data")
                st.dataframe(df.tail())

                # Feature & Target selection
                features = ['SMA', 'EMA', 'RSI', 'MACD', 'BB_H', 'BB_L', 'Volume_SMA']
                target = 'Target'

                # Placeholder for model (just showing columns here)
                st.subheader("📈 Model Training (Placeholder)")
                X = df[features]
                y = df[target]
                st.write("Feature sample:")
                st.dataframe(X.tail())
                st.write("Target sample:")
                st.dataframe(y.tail())

                st.info("You can now plug in any classifier (e.g., RandomForest, LogisticRegression).")

if __name__ == "__main__":
    main()
