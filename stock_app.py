import pandas as pd
import numpy as np
import ta  # Ensure you have the 'ta' (Technical Analysis) package installed
import yfinance as yf

# Function to fetch data
def fetch_data(symbol, start='2018-01-01', end='2024-12-31'):
    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        print(f"Data for {symbol} is empty or could not be fetched.")
    return df

# Function to compute technical indicators
def add_technical_indicators(df):
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

    # Compute each indicator
    for name, func in indicators.items():
        try:
            # Ensure the data passed is 1-dimensional (Series)
            df[name] = func()
            print(f"Successfully computed {name}")
        except Exception as e:
            print(f"Error computing {name}: {e}")

    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    # List of required columns
    required_cols = list(indicators.keys())
    available = [col for col in required_cols if col in df.columns]

    # Check for missing columns
    if len(available) < len(required_cols):
        missing = list(set(required_cols) - set(available))
        print(f"Missing columns: {missing}")

    # Only print NaNs for existing columns
    if available:
        print("NaN count before dropna:")
        print(df[available].isna().sum())

    # Drop rows with NaNs in the required indicators (only those that exist)
    df.dropna(subset=available, inplace=True)

    # Print the columns after adding indicators
    print(f"Dataframe columns after adding indicators: {df.columns}")

    # Check if DataFrame is empty after computation
    if df.empty:
        print("Warning: DataFrame is empty after adding technical indicators.")

    return df

# Function to train the model
def train_model(df):
    required_cols = ['SMA', 'EMA', 'RSI', 'MACD', 'BB_H', 'BB_L', 'Volume_SMA']
    
    # Check for missing columns before model training
    missing_columns = [col for col in required_cols if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns for model training: {missing_columns}")
    
    # Features and target
    X = df[required_cols]
    y = df['Target']
    
    # Example: Training a simple model (e.g., RandomForest)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Model accuracy
    acc = model.score(X, y)
    return model, acc

# Main code to execute the workflow
if __name__ == "__main__":
    symbol = 'WIPRO.NS'  # Example stock symbol
    df = fetch_data(symbol)
    
    if not df.empty:
        df = add_technical_indicators(df)
        
        # Print DataFrame columns before training
        print(f"Dataframe columns before model training: {df.columns}")
        
        # Train model
        try:
            model, acc = train_model(df)
            print(f"Model trained successfully with accuracy: {acc}")
        except ValueError as e:
            print(e)
    else:
        print("Data is empty. Model training will not proceed.")
