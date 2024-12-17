import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

def prepare_data(df, lookback=60):
    """
    Prepare data for LSTM model
    """
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i])
    
    return np.array(X), np.array(y), scaler

def create_model(lookback):
    """
    Create LSTM model
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    # Load data
    df = pd.read_csv('data/AAPL_processed.csv')
    
    # Prepare data
    X, y, scaler = prepare_data(df)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and train model
    model = create_model(lookback=60)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
    
    # Save model and scaler
    model.save('saved_models/lstm_model.h5')
    joblib.dump(scaler, 'saved_models/scaler.pkl')

if __name__ == "__main__":
    main() 