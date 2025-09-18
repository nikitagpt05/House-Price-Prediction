from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import sklearn  # Important for joblib to find scikit-learn objects

app = Flask(__name__)

# --- Load the Scaler and Model ---
# These files must be in the same directory as app.py
try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('xgb_model.pkl')
except FileNotFoundError as e:
    # This error will show in the Render logs if files are missing
    print(f"Error loading model files: {e}")
    scaler = None
    model = None

# --- Define the feature order the model expects ---
# This order MUST match the columns used when fitting the scaler in your notebook
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
    # Pass the options to the template for the dropdown menu
    return render_template('index.html', ocean_proximity_options=ocean_proximity_options)


@app.route('/predict', methods=['POST'])
def predict():
    # Defensive check in case model loading failed
    if not model or not scaler:
        return render_template('index.html', 
                               prediction_text="Error: Model or scaler not loaded. Check Render logs.",
                               ocean_proximity_options=ocean_proximity_options)
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

        # --- Preprocessing Steps (Must match the notebook EXACTLY) ---
        
        # 1. Apply log transformation
        log_cols = ['total_rooms', 'total_bedrooms', 'population', 'households']
        for col in log_cols:
            input_df[col] = np.log(input_df[col] + 1)
            
        # 2. Create the engineered features
        input_df['bedroom_ratio'] = input_df['total_bedrooms'] / input_df['total_rooms']
        input_df['household_rooms'] = input_df['total_rooms'] / input_df['households']

        # 3. Perform One-Hot Encoding for 'ocean_proximity'
        ocean_proximity_value = input_df.pop('ocean_proximity')

        for col in ocean_proximity_options:
            input_df[col] = 0
            
        if ocean_proximity_value.iloc[0] in input_df.columns:
            input_df[ocean_proximity_value.iloc[0]] = 1

        # 4. Ensure column order is correct before scaling
        # Use reindex to add any missing columns (like 'ISLAND') and ensure order
        final_df = input_df.reindex(columns=expected_columns, fill_value=0)
        
        # --- Prediction ---
        
        # Scale the features
        input_scaled = scaler.transform(final_df)

        # Make a prediction
        prediction = model.predict(input_scaled)
        
        # Format the prediction for display
        output = f"${prediction[0]:,.2f}"
        
        return render_template('index.html', 
                               prediction_text=output, 
                               ocean_proximity_options=ocean_proximity_options, 
                               form_data=form_data)

    except Exception as e:
        # Return a user-friendly error message on the web page
        error_message = f"An error occurred: {e}"
        return render_template('index.html', 
                               prediction_text=error_message, 
                               ocean_proximity_options=ocean_proximity_options)


if __name__ == "__main__":
    app.run(debug=True)