import yfinance as yf
import pandas as pd
import numpy as np
import ta
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Function to add technical indicators
def add_technical_indicators(df):
    if 'Close' not in df.columns or 'Volume' not in df.columns:
        st.error("Missing 'Close' or 'Volume' columns in the stock data.")
        return df  # Early exit if essential columns are missing
    
    close = df['Close']
    volume = df['Volume']

    indicators = {
        'SMA': lambda: ta.trend.sma_indicator(close=close, window=14),
        'EMA': lambda: ta.trend.ema_indicator(close=close, window=14),
        'RSI': lambda: ta.momentum.rsi(close=close, window=14),
        'MACD': lambda: ta.trend.macd_diff(close=close),
        'BB_H': lambda: ta.volatility.bollinger_hband(close=close),
        'BB_L': lambda: ta.volatility.bollinger_lband(close=close),
        'Volume_SMA': lambda: ta.trend.sma_indicator(close=volume, window=14)
    }

    for name, func in indicators.items():
        try:
            df[name] = func()
        except Exception as e:
            print(f"Error computing {name}: {e}")

    # Create target column (next day's price movement)
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    # Check if the required columns are present
    required_cols = list(indicators.keys())
    available = [col for col in required_cols if col in df.columns]

    if len(available) < len(required_cols):
        missing = list(set(required_cols) - set(available))
        st.write(f"Missing columns: {missing}")
    
    # Drop rows with NaN values in the available columns
    df.dropna(subset=available, inplace=True)

    # Print number of NaN values before dropping
    if available:
        st.write("NaN count before dropna:")
        st.write(df[available].isna().sum())

    return df

# Function to train a RandomForest model
def train_model(df):
    features = ['SMA', 'EMA', 'RSI', 'MACD', 'BB_H', 'BB_L', 'Volume_SMA']
    missing_columns = [col for col in features if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing columns for model training: {missing_columns}")

    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc, X_test

# Function to plot feature importance
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_

    # Sort the importance values and corresponding features
    sorted_idx = np.argsort(importance)
    sorted_importance = importance[sorted_idx]
    sorted_features = np.array(feature_names)[sorted_idx]

    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importance, color='royalblue')
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Model')
    plt.show()

# Streamlit UI
st.title('📊 Stock Technical Analysis & Prediction')

stock_ticker = st.text_input("Enter Stock Ticker (e.g., WIPRO.NS):")

if stock_ticker:
    st.write(f"Fetching data for: {stock_ticker}")
    
    try:
        # Fetch stock data from Yahoo Finance
        df = yf.download(stock_ticker, period="1y", interval="1d")
        
        if df.empty:
            st.error("Failed to fetch data. Please check the ticker.")
        else:
            # Add technical indicators to the dataframe
            df = add_technical_indicators(df)
            
            # Train the model and evaluate accuracy
            try:
                model, acc, X_test = train_model(df)
                st.write(f"Model accuracy: {acc:.2f}")

                # Plot feature importance
                st.subheader("Feature Importance")
                plot_feature_importance(model, X_test.columns)

            except ValueError as e:
                st.error(str(e))
                
    except Exception as e:
        st.error(f"Error fetching data: {e}")
