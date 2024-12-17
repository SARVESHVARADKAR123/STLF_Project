from flask import Flask, request, jsonify
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_utils import load_model_and_scaler, make_prediction
import numpy as np

app = Flask(__name__)

# Load model at startup
model, scaler = load_model_and_scaler()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data['input_data']).reshape(-1, 1)
        
        prediction = make_prediction(model, scaler, input_data)
        
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 400
            
        return jsonify({'prediction': float(prediction)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 