from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = 'model/xgboost_rent_model.pkl'

model_data = None

def load_model():
    global model_data
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        print(f"Model loaded successfully with {len(model_data['feature_names'])} features")
        return True
    else:
        print(f"Model file not found at {MODEL_PATH}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model_data is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        data = request.json
        
        features = [
            float(data.get('year', 2024)),
            float(data.get('community_id', 0)),
            float(data.get('properties', 0)),
            float(data.get('shape_area', 0)),
            float(data.get('shape_length', 0)),
            float(data.get('mixed_rate', 0)),
            float(data.get('area_encoded', 0)),
            float(data.get('cost_category_encoded', 0)),
            float(data.get('change_category_encoded', 0)),
            float(data.get('properties_per_area', 0)),
            float(data.get('year_since_2000', 0))
        ]
        
        features_df = pd.DataFrame([features], columns=model_data['feature_names'])
        
        prediction = model_data['model'].predict(features_df)[0]
        prediction = float(prediction)
        
        return jsonify({
            'success': True,
            'predicted_rent': round(prediction, 2),
            'formatted_rent': f"${prediction:,.2f}"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_data is not None
    })

if __name__ == '__main__':
    if load_model():
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Please train the model first by running: python xgboost_price_predictor.py")
