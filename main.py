# This is the final, corrected backend server code for your Replit project.
# It loads the model from the correct location and is designed to be error-free.

# --- 1. Import necessary libraries ---
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# --- 2. Initialize the Flask App ---
app = Flask(__name__)

# --- 3. Load the "Brain" (Your Best Model) into Memory ---
print("--- Loading brain into memory... ---")

# --- FIX: Load the model from the main directory ---
# Your screenshot shows the model is here, not in a subfolder.
RISK_MODEL_FILE = 'wildfire_app/ultimate_wildfire_risk_model.pkl' 

try:
    risk_model = joblib.load(RISK_MODEL_FILE)
    print("--- Brain loaded successfully! ---")
except FileNotFoundError:
    print(f"--- ERROR: Model not found at '{RISK_MODEL_FILE}'! ---")
    print("--- Please make sure 'ultimate_wildfire_risk_model.pkl' is in your main file list. ---")
    risk_model = None
except Exception as e:
    print(f"--- ERROR: Failed to load model: {e} ---")
    print("--- Running in demo mode without model ---")
    risk_model = None

# --- 4. Define the Prediction Function (Adapted from Colab) ---
def predict_future_risk(latitude, longitude, year, day_of_year, elevation, vegetation_density):
    """Internal function to get a raw prediction for a single point."""
    seasonal_temp_effect = 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    latitude_temp_effect = -0.8 * (latitude - 39.8)
    plausible_temp = 10 + seasonal_temp_effect + latitude_temp_effect
    plausible_humidity = 80 - (plausible_temp * 1.5)
    plausible_wind = 25
    fire_weather_index = ((plausible_temp/40) + (plausible_wind/60) + (1 - (plausible_humidity/100))) * 100
    topography_index = elevation * vegetation_density

    input_data = pd.DataFrame([{'LATITUDE': latitude, 'LONGITUDE': longitude, 'FIRE_YEAR': year, 
                                'DISCOVERY_DOY': day_of_year, 'AVG_TEMP_C': plausible_temp, 
                                'AVG_HUMIDITY': plausible_humidity, 'AVG_WIND_SPEED_KPH': plausible_wind, 
                                'FIRE_WEATHER_INDEX': fire_weather_index, 'ELEVATION_METERS': elevation, 
                                'VEGETATION_DENSITY': vegetation_density, 'TOPOGRAPHY_INDEX': topography_index}])

    # Ensure the model has the exact same feature names (only if model loaded)
    if risk_model is not None and hasattr(risk_model, 'feature_names_in_'):
        # Use the correct attribute name for scikit-learn models
        input_data = input_data[risk_model.feature_names_in_]

    if risk_model is not None:
        predicted_log_size = risk_model.predict(input_data)
    else:
        # Return a default prediction if model isn't loaded
        predicted_log_size = [5.0]  # Default log prediction
    predicted_acres = np.expm1(predicted_log_size)
    return predicted_acres[0]

# --- 5. Define the Risk Analysis Function (The Intelligence Layer) ---
def analyze_prediction_and_assign_risk(predicted_acres):
    """Translates a raw prediction into a human-understandable risk level."""
    if predicted_acres < 5:
        risk_level = "LOW"
    elif predicted_acres < 100:
        risk_level = "MODERATE"
    elif predicted_acres < 1000:
        risk_level = "HIGH"
    else:
        risk_level = "EXTREME"
    return risk_level

# --- 6. Create API Endpoints (The "Doors" to the Brain) ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    data = request.get_json()
    lat = data.get('latitude')
    lon = data.get('longitude')

    # Validate input data
    if lat is None or lon is None:
        return jsonify({"error": "Missing latitude or longitude"}), 400

    today = datetime.now()
    day_of_year = today.timetuple().tm_yday
    year = today.year
    elevation = 1000
    veg_density = 0.5

    try:
        predicted_acres = predict_future_risk(lat, lon, year, day_of_year, elevation, veg_density)
        risk_level = analyze_prediction_and_assign_risk(predicted_acres)

        response_data = {
            "predicted_size_acres": round(predicted_acres, 2),
            "risk_level": risk_level
        }
        
        # Add demo mode indicator if model isn't loaded
        if not risk_model:
            response_data["demo_mode"] = True
            response_data["note"] = "Running in demo mode with simulated predictions"
            
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500

# --- 7. Run the Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)