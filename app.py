"""
F1 Race Winner Prediction API
A Flask backend that serves predictions from our trained ML model.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__, template_folder='webapp', static_folder='webapp/static')

# Load our trained model and preprocessing objects
try:
    model_data = joblib.load('out/models/f1_model_data.pkl')
    model = model_data['model']
    driver_encoder = model_data['le_driver']
    team_encoder = model_data['le_constructor']
    circuit_encoder = model_data['le_circuit']
    scaler = model_data.get('scaler')
    feature_names = model_data['feature_names']
    
    print(f"✓ Successfully loaded {model_data.get('model_name', 'model')}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
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
    (drivers, teams, circuits from our dataset)
    """
    return jsonify({
        'drivers': sorted(list(driver_encoder.classes_)),
        'constructors': sorted(list(team_encoder.classes_)),
        'circuits': sorted(list(circuit_encoder.classes_))
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
        current_points = data['points']
        
        # Encode categorical variables (convert names to numbers)
        # If we get something unknown, default to -1
        driver_encoded = driver_encoder.transform([driver])[0] if driver in driver_encoder.classes_ else -1
        team_encoded = team_encoder.transform([team])[0] if team in team_encoder.classes_ else -1
        circuit_encoded = circuit_encoder.transform([circuit])[0] if circuit in circuit_encoder.classes_ else -1
        
        # Build the feature set that matches our training data
        # Note: We're approximating driver age at 28 since we don't ask for DOB
        features_df = pd.DataFrame([{
            'grid': grid_position,
            'points': current_points,
            'laps': num_laps,
            'driverId_enc': driver_encoded,
            'constructorId_enc': team_encoded,
            'circuitId_enc': circuit_encoded,
            'driver_age': 28  # Default assumption
        }])
        
        # Make sure features are in the right order
        features_df = features_df[feature_names]
        
        # Scale the numerical features if we have a scaler
        if scaler:
            numerical_cols = ['grid', 'points', 'laps', 'driver_age']
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
