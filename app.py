import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import joblib

# Initialize the flask app
app = Flask(__name__)

# --- Load the trained XGBoost model and scaler ---
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    model = None
    scaler = None

# Define the expected order of columns for the model
# This must match the order of columns used during training
EXPECTED_COLUMNS = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN',
    'bedroom_ratio', 'household_rooms'
]

@app.route('/')
def home():
    """Render the home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receive user input, preprocess it, and return the XGBoost model's prediction."""
    
    if not model or not scaler:
        return render_template('index.html', 
                               prediction_text='Error: Model/scaler not loaded. Check server logs.')

    try:
        # --- Get the input features from the form ---
        input_features = {
            "longitude": float(request.form['longitude']),
            "latitude": float(request.form['latitude']),
            "housing_median_age": float(request.form['housing_median_age']),
            "total_rooms": float(request.form['total_rooms']),
            "total_bedrooms": float(request.form['total_bedrooms']),
            "population": float(request.form['population']),
            "households": float(request.form['households']),
            "median_income": float(request.form['median_income']),
            "ocean_proximity": request.form['ocean_proximity']
        }
        input_df = pd.DataFrame([input_features])

        # --- Preprocessing Steps (replicated from your notebook) ---

        # 1. Log Transformation
        for col in ['total_rooms', 'total_bedrooms', 'population', 'households']:
            input_df[col] = np.log(input_df[col] + 1)

        # 2. One-Hot Encode 'ocean_proximity'
        ocean_prox_dummies = pd.get_dummies(input_df['ocean_proximity'], prefix='').astype(int)
        input_df = pd.concat([input_df.drop('ocean_proximity', axis=1), ocean_prox_dummies], axis=1)

        # 3. Add missing dummy columns to match training set
        for col in ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']:
            if col not in input_df.columns:
                input_df[col] = 0

        # 4. Feature Engineering
        input_df['bedroom_ratio'] = input_df['total_bedrooms'] / input_df['total_rooms']
        input_df['household_rooms'] = input_df['total_rooms'] / input_df['households']
        
        # 5. Ensure columns are in the correct order
        input_df = input_df[EXPECTED_COLUMNS]

        # 6. Scale the features using the loaded scaler
        input_scaled = scaler.transform(input_df)

        # --- Make Prediction ---
        prediction = model.predict(input_scaled)
        
        output_text = f'Predicted House Price: ${prediction[0]:,.2f}'

    except Exception as e:
        output_text = f'An error occurred: {e}'

    return render_template('index.html', prediction_text=output_text)

if __name__ == '__main__':
    app.run(debug=True)