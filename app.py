from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import sklearn

app = Flask(__name__)

# --- Load Models and Scaler ---
# Make sure to update these file paths if they are different
try:
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    # This is a fallback for local testing if the notebook is run first
    # In a production environment, these files should be pre-built
    from HousePricePrediction import xgb as model, scaler

# --- Define the feature order the model expects ---
# This order MUST match the columns used when fitting the scaler
expected_columns = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms', 
    'total_bedrooms', 'population', 'households', 'median_income',
    'bedroom_ratio', 'household_rooms', '<1H OCEAN', 'INLAND', 
    'ISLAND', 'NEAR BAY', 'NEAR OCEAN'
]
# Define the categorical options for the dropdown in the HTML
ocean_proximity_options = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']


@app.route('/')
def home():
    # Pass the options to the template
    return render_template('index.html', ocean_proximity_options=ocean_proximity_options)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- Get data from form ---
        form_data = request.form.to_dict()
        
        # Convert numeric fields from string to float
        numeric_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                            'total_bedrooms', 'population', 'households', 'median_income']
        for feature in numeric_features:
            form_data[feature] = float(form_data[feature])

        # Create a pandas DataFrame from the form data
        input_df = pd.DataFrame([form_data])

        # --- Preprocessing Steps (Must match the notebook) ---
        
        # 1. Apply log transformation to the same columns as in training
        log_cols = ['total_rooms', 'total_bedrooms', 'population', 'households']
        for col in log_cols:
            input_df[col] = np.log(input_df[col] + 1)
            
        # 2. Create the engineered features
        input_df['bedroom_ratio'] = input_df['total_bedrooms'] / input_df['total_rooms']
        input_df['household_rooms'] = input_df['total_rooms'] / input_df['households']

        # 3. Perform One-Hot Encoding for 'ocean_proximity'
        # Get the value from the form
        ocean_proximity_value = input_df.pop('ocean_proximity')

        # Add all possible ocean_proximity columns and set to 0
        for col in ocean_proximity_options:
            input_df[col] = 0
            
        # Set the selected ocean_proximity column to 1
        if ocean_proximity_value[0] in input_df.columns:
            input_df[ocean_proximity_value[0]] = 1

        # 4. Ensure the column order matches the training data
        input_df = input_df.reindex(columns=expected_columns)
        
        # --- Prediction ---
        
        # Scale the data using the loaded scaler
        input_scaled = scaler.transform(input_df)

        # Make a prediction
        prediction = model.predict(input_scaled)
        
        # Format the prediction for display
        output = f"${prediction[0]:,.2f}"
        
        return render_template('index.html', prediction_text=output, ocean_proximity_options=ocean_proximity_options)

    except Exception as e:
        # Return a user-friendly error message
        error_message = f"An error occurred: {e}"
        return render_template('index.html', prediction_text=error_message, ocean_proximity_options=ocean_proximity_options)


if __name__ == "__main__":
    app.run(debug=True)