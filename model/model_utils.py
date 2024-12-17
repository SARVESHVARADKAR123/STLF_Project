import numpy as np
from tensorflow.keras.models import load_model
import joblib

def load_model_and_scaler():
    """
    Load the trained model and scaler
    """
    try:
        model = load_model('saved_models/lstm_model.h5')
        scaler = joblib.load('saved_models/scaler.pkl')
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def make_prediction(model, scaler, input_data):
    """
    Make predictions using the loaded model
    """
    if model is None or scaler is None:
        return None
    
    # Scale input data
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(np.array([scaled_input]))
    
    # Inverse transform prediction
    prediction = scaler.inverse_transform(prediction)
    
    return prediction[0][0] 