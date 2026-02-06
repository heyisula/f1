"""
F1 Race Winner Prediction API
A Flask backend that serves predictions from our trained ML model.
"""

import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='webapp', static_folder='webapp/static')

# Loading trained model and preprocessing objects
try:
    model_data = joblib.load('out/models/f1_model_data.pkl')
    model = model_data['model']
    driver_encoder = model_data['le_driver']
    team_encoder = model_data['le_constructor']
    circuit_encoder = model_data['le_circuit']
    scaler = model_data.get('scaler')
    feature_names = model_data['feature_names']
    
    print(f"[OK] Successfully loaded {model_data.get('model_name', 'model')}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    print("Make sure you've run train.ipynb first!")
    exit(1)


@app.route('/')
def home():
    """Serve the main prediction interface"""
    return render_template('index.html')


@app.route('/meta', methods=['GET'])
def get_dropdown_data():
    """
    Get all available options for the dropdowns
    (drivers, teams, circuits from our dataset) plus circuit metadata
    """
    # Define typical lap counts for circuits
    circuit_laps = {
        "adelaide": 82, "ain-diab": 53, "aintree": 90, "albert_park": 58, "americas": 56,
        "anderstorp": 80, "avus": 60, "bahrain": 57, "baku": 51, "boavista": 55,
        "brands_hatch": 80, "bremgarten": 66, "buddh": 60, "catalunya": 66, "charade": 40,
        "dallas": 67, "detroit": 63, "dijon": 80, "donington": 76, "essarts": 77,
        "estoril": 71, "fuji": 73, "galvez": 100, "george": 85, "hockenheimring": 67,
        "hungaroring": 70, "imola": 63, "indianapolis": 73, "interlagos": 71, "istanbul": 58,
        "jacarepagua": 61, "jarama": 90, "jeddah": 50, "jerez": 69, "kyalami": 80,
        "las_vegas": 50, "lemans": 80, "long_beach": 80, "losail": 57, "magny_cours": 72,
        "marina_bay": 61, "miami": 57, "monaco": 78, "monsanto": 62, "montjuic": 75,
        "monza": 53, "mosport": 90, "mugello": 59, "nivelles": 85, "nurburgring": 67,
        "okayama": 83, "pedralbes": 80, "pescara": 18, "phoenix": 81, "portimao": 66,
        "red_bull_ring": 71, "reims": 50, "ricard": 80, "riverside": 75, "rodriguez": 71,
        "sebring": 42, "sepang": 56, "shanghai": 56, "silverstone": 52, "sochi": 53,
        "spa": 44, "suzuka": 53, "tremblant": 90, "valencia": 57, "vegas": 50,
        "villeneuve": 70, "watkins_glen": 110, "yas_marina": 58, "yeongam": 55,
        "zandvoort": 72, "zeltweg": 54, "zolder": 70
    }

    return jsonify({
        'drivers': sorted(list(driver_encoder.classes_)),
        'constructors': sorted(list(team_encoder.classes_)),
        'circuits': sorted(list(circuit_encoder.classes_)),
        'circuit_laps': circuit_laps,
        'verified_features': feature_names
    })


@app.route('/predict', methods=['POST'])
def predict_winner():
    """
    Main prediction endpoint
    Takes race parameters and returns win probability
    """
    data = request.json
    
    try:
        # Extract input parameters
        driver = data['driver']
        team = data['constructor']
        circuit = data['circuit']
        grid_position = data['grid']
        num_laps = data['laps']
        current_points = data['points']  # This is cumulative points BEFORE the race
        
        # Validate categorical inputs
        if driver not in driver_encoder.classes_:
            return jsonify({'error': f'Unknown driver: {driver}. Available drivers can be fetched from /meta endpoint.'}), 400
        if team not in team_encoder.classes_:
            return jsonify({'error': f'Unknown team: {team}. Available teams can be fetched from /meta endpoint.'}), 400
        if circuit not in circuit_encoder.classes_:
            return jsonify({'error': f'Unknown circuit: {circuit}. Available circuits can be fetched from /meta endpoint.'}), 400
        
        # Validate and convert numeric inputs
        try:
            grid_position = int(grid_position)
            num_laps = int(num_laps)
            current_points = float(current_points)
            
            if not (1 <= grid_position <= 20):
                return jsonify({'error': 'Grid position must be between 1 and 20'}), 400
            if num_laps < 0:
                return jsonify({'error': 'Number of laps must be positive'}), 400
            if current_points < 0:
                return jsonify({'error': 'Points must be non-negative'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid numeric input'}), 400
        
        # Encoding categorical variables (convert names to numbers)
        driver_encoded = driver_encoder.transform([driver])[0]
        team_encoded = team_encoder.transform([team])[0]
        circuit_encoded = circuit_encoder.transform([circuit])[0]
        
        # Building the feature set that matches our training data
        # Note: 'cumulative_points' is pre-race points (fixes data leakage)
        # Note: We're approximating driver age at 28 since we don't ask for DOB
        features_df = pd.DataFrame([{
            'grid': grid_position,
            'cumulative_points': current_points,  # RENAMED from 'points' to fix data leakage
            'laps': num_laps,
            'driverId_enc': driver_encoded,
            'constructorId_enc': team_encoded,
            'circuitId_enc': circuit_encoded,
            'driver_age': 28  # Default assumption (average F1 driver age)
        }])
        
        # Making sure features are in the right order
        features_df = features_df[feature_names]
        
        # Scaling the numerical features if we have a scaler
        if scaler:
            numerical_cols = ['grid', 'cumulative_points', 'laps', 'driver_age']
            features_df[numerical_cols] = scaler.transform(features_df[numerical_cols])
            
        # Get the prediction!
        win_probability = model.predict_proba(features_df)[0][1]
        will_win = int(model.predict(features_df)[0]) == 1
        
        return jsonify({
            'probability': float(win_probability),
            'is_winner': bool(will_win)
        })

    except Exception as e:
        # Something went wrong - send back a helpful error
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 400


if __name__ == '__main__':
    print("     Starting F1 Predictor API...")
    print("     Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True)
