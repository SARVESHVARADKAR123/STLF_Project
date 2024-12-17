import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import os

def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance
    """
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the fetched data
    """
    if df is None:
        return None
    
    # Calculate technical indicators
    df['Returns'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Remove NaN values
    df = df.dropna()
    
    return df

def ensure_data_directory():
    """
    Create data directory if it doesn't exist
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir

def main():
    # Configuration
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]  # Example stock symbols
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years of data
    
    data_dir = ensure_data_directory()
    
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        # Fetch and preprocess data
        raw_data = fetch_stock_data(symbol, start_date, end_date)
        processed_data = preprocess_data(raw_data)
        
        # Save processed data
        if processed_data is not None:
            output_path = os.path.join(data_dir, f'{symbol}_processed.csv')
            processed_data.to_csv(output_path)
            print(f"Saved processed data to {output_path}")
        else:
            print(f"Failed to process data for {symbol}")

if __name__ == "__main__":
    main() 