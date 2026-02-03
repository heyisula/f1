from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Model
try:
    model_data = joblib.load('models/best_f1_model.pkl')
    model = model_data['model']
    le_driver = model_data['le_driver']
    le_constructor = model_data['le_constructor']
    le_circuit = model_data['le_circuit']
    scaler = model_data.get('scaler')
    feature_names = model_data['feature_names']
    print(f"Flask App Loaded Model: {model_data.get('model_name', 'Unknown')}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/meta', methods=['GET'])
def get_meta():
    """Return lists of drivers, constructors, circuits for the frontend dropdowns"""
    return jsonify({
        'drivers': sorted(list(le_driver.classes_)),
        'constructors': sorted(list(le_constructor.classes_)),
        'circuits': sorted(list(le_circuit.classes_))
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    try:
        driver = data['driver']
        constructor = data['constructor']
        circuit = data['circuit']
        grid = data['grid']
        laps = data['laps']
        points = data['points']
        
        # Encoding
        # Handle unknown categories gracefully (fallback to first class or -1)
        driver_enc = le_driver.transform([driver])[0] if driver in le_driver.classes_ else -1
        constructor_enc = le_constructor.transform([constructor])[0] if constructor in le_constructor.classes_ else -1
        circuit_enc = le_circuit.transform([circuit])[0] if circuit in le_circuit.classes_ else -1
        
        # Prepare DataFrame
        # 'grid', 'points', 'laps', 'driverId_enc', 'constructorId_enc', 'circuitId_enc', 'driver_age'
        # Approximating age as 28 since we don't ask for user DOB in basic UI
        input_df = pd.DataFrame([{
            'grid': grid,
            'points': points,
            'laps': laps,
            'driverId_enc': driver_enc,
            'constructorId_enc': constructor_enc,
            'circuitId_enc': circuit_enc,
            'driver_age': 28 
        }])
        
        # Ensure order
        input_df = input_df[feature_names]
        
        # Scale
        if scaler:
            cols_to_scale = ['grid', 'points', 'laps', 'driver_age']
            input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
            
        # Predict
        prob = model.predict_proba(input_df)[0][1] # Probability of winning
        prediction = int(model.predict(input_df)[0])
        
        return jsonify({
            'probability': float(prob),
            'is_winner': bool(prediction == 1)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
