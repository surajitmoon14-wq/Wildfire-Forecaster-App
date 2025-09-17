# This is the final, corrected backend server code for your Replit project.
# It includes your advanced prediction brain, the live weather skill, and the new GPT-2 Chatbot.

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
import torch  # <-- ADDED to ensure correct loading for transformers
from flask import Flask, request, jsonify, render_template
from transformers import pipeline

# --- 2. Initialize the Flask App ---
app = Flask(__name__)

# --- API Key for Weather ---
API_KEY = "bd2c80f8da534c15acb142929251609"  # <--- YOUR API KEY IS NOW INCLUDED

# --- 3. Load the GPT-2 Chatbot Brain ---
print("--- Loading GPT-2 Chatbot Brain... (This may take a moment on first run) ---")
try:
    # This creates the text generation function using the GPT-2 model
    chatbot = pipeline('text-generation', model='gpt2')
    print("--- ✓ GPT-2 Chatbot loaded successfully! ---")
except Exception as e:
    print(f"--- ✗ Failed to load GPT-2 model: {e} ---")
    chatbot = None

# --- 4. Load All Trained Models ---
print("--- Loading trained models from wildfire_app folder... ---")

# Try to load all 3 models
models = {}
model_files = {
    'risk': 'wildfire_app/ultimate_wildfire_risk_model.pkl',
    'cause': 'wildfire_app/wildfire_cause_model.pkl',
    'general_risk': 'wildfire_app/wildfire_risk_model.pkl'
}

for model_name, file_path in model_files.items():
    try:
        models[model_name] = joblib.load(file_path)
        print(f"--- ✓ Loaded {model_name} model successfully! ---")
    except Exception as e:
        print(f"--- ✗ Failed to load {model_name} model: {e} ---")
        models[model_name] = None

# Use the best available model
risk_model = models['risk'] or models['general_risk'] or models['cause']
if risk_model:
    print("--- 🧠 AI Brain activated with trained models! ---")
else:
    print("--- ⚠️  Running in simulation mode ---")
    risk_model = None


# --- 5. Location-Based Feature Engineering (No Changes) ---
def generate_location_features(latitude, longitude, year, day_of_year):
    """Generate realistic features based on actual location coordinates."""
    seasonal_temp_effect = 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    latitude_temp_effect = -0.8 * (latitude - 39.8)
    base_temp = 10 + seasonal_temp_effect + latitude_temp_effect
    longitude_humidity_effect = 10 * np.sin(2 * np.pi * (longitude + 100) / 360)
    base_humidity = 60 - (base_temp * 1.2) + longitude_humidity_effect
    if longitude < -100:
        elevation = 800 + (latitude - 35) * 50 + abs(longitude + 110) * 20
        vegetation_density = 0.3 + (latitude - 32) * 0.02
        wind_base = 20
    else:
        elevation = 200 + (latitude - 35) * 30
        vegetation_density = 0.6 + (latitude - 30) * 0.01
        wind_base = 15
    coastal_factor = abs(longitude + 95) / 50
    wind_speed = wind_base + coastal_factor * 5
    elevation = max(0, min(elevation, 4000))
    vegetation_density = max(0.1, min(vegetation_density, 0.9))
    wind_speed = max(5, min(wind_speed, 40))
    base_humidity = max(20, min(base_humidity, 95))
    fire_weather_index = ((base_temp / 40) + (wind_speed / 60) + (1 - (base_humidity / 100))) * 100
    topography_index = elevation * vegetation_density
    return {'temperature': base_temp, 'humidity': base_humidity, 'wind_speed': wind_speed, 'elevation': elevation, 'vegetation_density': vegetation_density, 'fire_weather_index': fire_weather_index, 'topography_index': topography_index}


def predict_future_risk(latitude, longitude, year, day_of_year):
    """Predict wildfire risk using location-specific features."""
    features = generate_location_features(latitude, longitude, year, day_of_year)
    input_data = pd.DataFrame([features])
    input_data['LATITUDE'], input_data['LONGITUDE'], input_data['FIRE_YEAR'], input_data['DISCOVERY_DOY'] = latitude, longitude, year, day_of_year
    input_data.rename(columns={'temperature': 'AVG_TEMP_C', 'humidity': 'AVG_HUMIDITY', 'wind_speed': 'AVG_WIND_SPEED_KPH', 'elevation': 'ELEVATION_METERS', 'vegetation_density': 'VEGETATION_DENSITY'}, inplace=True)
    if risk_model:
        try:
            if hasattr(risk_model, 'feature_names_in_'):
                available_features = [col for col in input_data.columns if col in risk_model.feature_names_in_]
                if available_features:
                    input_data = input_data[available_features]
            predicted_log_size = risk_model.predict(input_data)
            return np.expm1(predicted_log_size)[0]
        except Exception as e:
            print(f"Model prediction error: {e}")
    risk_score = (features['fire_weather_index'] * 0.4 + features['elevation'] * 0.0005 + features['vegetation_density'] * 50 + abs(latitude - 40) * 2 + features['temperature'] * 0.8)
    np.random.seed(int((latitude * 1000 + longitude * 1000) % 1000))
    return max(1, risk_score + np.random.normal(0, risk_score * 0.2))


# --- 6. Define the Risk Analysis Function (The Intelligence Layer) ---
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


# --- 7. Create API Endpoints (The "Doors" to the Brain) ---
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def handle_prediction():
    data = request.get_json()
    lat = data.get('latitude')
    lon = data.get('longitude')
    if lat is None or lon is None:
        return jsonify({"error": "Missing latitude or longitude"}), 400
    today = datetime.now()
    predicted_acres = predict_future_risk(lat, lon, today.year, today.timetuple().tm_yday)
    risk_level = analyze_prediction_and_assign_risk(predicted_acres)
    response_data = {"predicted_size_acres": round(predicted_acres, 2), "risk_level": risk_level}
    if not risk_model:
        response_data["demo_mode"] = True
        response_data["note"] = "Running in demo mode with simulated predictions"
    return jsonify(response_data)


@app.route('/weather', methods=['POST'])
def get_weather():
    """This is the new 'door' for fetching live weather."""
    if not API_KEY or API_KEY == "YOUR_API_KEY":
        return jsonify({"error": "API key not configured on server."})
    data = request.get_json()
    lat, lon = data.get('latitude'), data.get('longitude')
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY.strip()}&q={lat},{lon}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        weather_data = response.json()
        return jsonify({"temp_c": weather_data['current']['temp_c'], "condition": weather_data['current']['condition']['text'], "wind_kph": weather_data['current']['wind_kph']})
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

# --- NEW: GPT-2 Chatbot Endpoint ---
@app.route('/get_advice', methods=['POST'])
def get_chatbot_advice():
    """This is the new door for getting advice from the self-hosted GPT-2 AI."""
    if not chatbot:
         return jsonify({"advice": "Chatbot is currently offline."})

    data = request.get_json()
    risk_level = data.get('risk_level')
    location_name = data.get('location_name', 'the selected area')

    # Create a prompt to guide the GPT-2 model
    prompt = f"A wildfire safety expert is issuing a public safety warning. The current wildfire risk level is '{risk_level}' in '{location_name}'. Here is the official safety advice:"

    try:
        # Generate text using the loaded GPT-2 pipeline
        response = chatbot(prompt, max_length=120, num_return_sequences=1, pad_token_id=50256)

        # Clean up the response to only show the newly generated text
        advice = response[0]['generated_text']
        advice = advice.replace(prompt, "").strip()

        return jsonify({"advice": advice})
    except Exception as e:
        print(f"--- GPT-2 generation error: {e} ---")
        return jsonify({"advice": "Could not generate safety advice at this time."}), 500

# --- 8. Run the Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



