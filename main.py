# This is the final backend server code, adjusted for your file structure.

# --- 1. Import necessary libraries ---
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os # To build file paths correctly

# --- 2. Initialize the Flask App ---
app = Flask(__name__)

# --- 3. Load the "Brain" (Your Best Model) into Memory ---
print("--- Loading brain into memory... ---")

# --- IMPORTANT: We build the path to the models inside the 'wildfire_app' folder ---
MODEL_DIR = 'wildfire_app'
RISK_MODEL_FILE = os.path.join(MODEL_DIR, 'ultimate_wildfire_risk_model.pkl')

try:
    risk_model = joblib.load(RISK_MODEL_FILE)
    print("--- Brain loaded successfully! ---")
except FileNotFoundError:
    print(f"--- ERROR: Model not found at '{RISK_MODEL_FILE}'! ---")
    print("--- Please make sure the 'ultimate_wildfire_risk_model.pkl' file is inside the 'wildfire_app' folder. ---")
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

    predicted_log_size = risk_model.predict(input_data)
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
    if not risk_model:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    lat = data.get('latitude')
    lon = data.get('longitude')

    today = datetime.now()
    day_of_year = today.timetuple().tm_yday
    year = today.year
    elevation = 1000  # Example value
    veg_density = 0.5 # Example value

    predicted_acres = predict_future_risk(lat, lon, year, day_of_year, elevation, veg_density)
    risk_level = analyze_prediction_and_assign_risk(predicted_acres)

    return jsonify({
        "predicted_size_acres": round(predicted_acres, 2),
        "risk_level": risk_level
    })

# --- 7. Run the Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81)

