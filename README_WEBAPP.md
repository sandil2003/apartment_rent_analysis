# Apartment Rent Predictor Web Application

A modern, responsive web application for predicting apartment rent prices using XGBoost machine learning model.

## Features

- **AI-Powered Predictions**: Uses XGBoost regression model trained on real apartment market data
- **Modern UI**: Beautiful gradient design with Tailwind CSS
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Interactive Interface**: Real-time form validation and smooth animations
- **Easy to Use**: Simple form-based input with instant predictions

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure the model is trained (the model file should exist at `model/xgboost_rent_model.pkl`)

2. Start the Flask server:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## How to Use

1. Fill in the apartment details in the form:
   - **Year**: The year for prediction (e.g., 2024)
   - **Community Area ID**: Numeric identifier for the community area
   - **Number of Properties**: Total properties in the tract
   - **Shape Area**: Geographic area size
   - **Shape Length**: Perimeter length
   - **Mixed-Rate Apartments**: Whether the tract has mixed-rate apartments
   - **Area Encoded**: Encoded community area name
   - **Cost Category Encoded**: Encoded cost category
   - **Change Category Encoded**: Encoded year-over-year change category

2. Click "Predict Rent" button

3. View the predicted monthly rent displayed on screen

## Technology Stack

- **Backend**: Flask (Python web framework)
- **ML Model**: XGBoost Regressor
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Icons**: Font Awesome
- **Fonts**: Google Fonts (Inter)

## API Endpoints

### GET /
Returns the main web interface

### POST /predict
Predicts apartment rent based on input features

**Request Body** (JSON):
```json
{
  "year": 2024,
  "community_id": 1,
  "properties": 100,
  "shape_area": 1000000,
  "shape_length": 5000,
  "mixed_rate": 0,
  "area_encoded": 0,
  "cost_category_encoded": 1,
  "change_category_encoded": 0,
  "properties_per_area": 0.0001,
  "year_since_2000": 24
}
```

**Response** (JSON):
```json
{
  "success": true,
  "predicted_rent": 1234.56,
  "formatted_rent": "$1,234.56"
}
```

### GET /health
Returns application health status

## Model Performance

- **Test R² Score**: 0.9844
- **Test MAE**: $35.65
- **Test RMSE**: $51.66

## Files Structure

```
apartment_rent_analysis/
├── app.py                          # Flask application
├── templates/
│   └── index.html                  # Main web interface
├── static/
│   └── script.js                   # Frontend JavaScript
├── model/
│   └── xgboost_rent_model.pkl     # Trained ML model
├── xgboost_price_predictor.py     # Model training script
└── requirements.txt                # Python dependencies
```

## Troubleshooting

**Model not found error:**
- Run `python xgboost_price_predictor.py` to train and save the model first

**Port already in use:**
- Change the port in `app.py` by modifying `app.run(port=5000)` to a different port number

**Module not found errors:**
- Install missing dependencies: `pip install -r requirements.txt`
