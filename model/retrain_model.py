import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from train_model import prepare_data

def retrain_model(model, X_new, y_new, epochs=10):
    """
    Retrain existing model with new data
    """
    history = model.fit(X_new, y_new, epochs=epochs, batch_size=32, validation_split=0.1)
    return model, history

def main():
    # Load existing model and scaler
    model = load_model('saved_models/lstm_model.h5')
    scaler = joblib.load('saved_models/scaler.pkl')
    
    # Load new data
    new_data = pd.read_csv('data/AAPL_processed.csv')
    
    # Prepare new data
    X_new, y_new, _ = prepare_data(new_data)
    
    # Retrain model
    model, history = retrain_model(model, X_new, y_new)
    
    # Save updated model
    model.save('saved_models/lstm_model.h5')

if __name__ == "__main__":
    main() 