import gradio as gr
import pandas as pd
import joblib
import numpy as np

# Load the trained model and helper objects
try:
    model_data = joblib.load('models/best_f1_model.pkl')
    model = model_data['model']
    le_driver = model_data['le_driver']
    le_constructor = model_data['le_constructor']
    le_circuit = model_data['le_circuit']
    scaler = model_data.get('scaler') # Get scaler if it exists
    feature_names = model_data['feature_names']
    print(f"Loaded model: {model_data.get('model_name', 'Unknown')}")
except FileNotFoundError:
    print("Model file not found. Please run train.ipynb first.")
    exit()

def predict_winner(grid_position, points, driver_name, constructor_name, circuit_name, laps):
    # Create a dataframe for the input
    try:
        # Encode inputs
        driver_id = le_driver.transform([driver_name])[0] if driver_name in le_driver.classes_ else -1
        constructor_id = le_constructor.transform([constructor_name])[0] if constructor_name in le_constructor.classes_ else -1
        circuit_id = le_circuit.transform([circuit_name])[0] if circuit_name in le_circuit.classes_ else -1
        
        # We need to construct the feature vector matching the training data
        input_data = pd.DataFrame({
            'grid': [grid_position],
            'points': [points],
            'laps': [laps],
            'driverId_enc': [driver_id],
            'constructorId_enc': [constructor_id],
            'circuitId_enc': [circuit_id],
            'driver_age': [30] # approximated as mean age, or could add another input
        })
        
        # Ensure column order matches training
        input_data = input_data[feature_names]
        
        # Scale numerical features if scaler exists
        if scaler:
            cols_to_scale = ['grid', 'points', 'laps', 'driver_age']
            input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])
        
        # Predict probability
        prob = model.predict_proba(input_data)[0][1] # Probability of class 1 (Winner)
        prediction = model.predict(input_data)[0]
        
        result_text = "Winner!" if prediction == 1 else "Not a Winner"
        return f"Prediction: {result_text} (Win Probability: {prob:.2%})"
        
    except Exception as e:
        return f"Error making prediction: {str(e)}"

# Get lists for dropdowns
drivers_list = list(le_driver.classes_)
constructors_list = list(le_constructor.classes_)
circuits_list = list(le_circuit.classes_)

# Create Gradio Interface
iface = gr.Interface(
    fn=predict_winner,
    inputs=[
        gr.Slider(1, 20, step=1, label="Grid Start Position"),
        gr.Number(label="Current Points", value=0),
        gr.Dropdown(drivers_list, label="Driver"),
        gr.Dropdown(constructors_list, label="Constructor (Team)"),
        gr.Dropdown(circuits_list, label="Circuit"),
        gr.Number(label="Laps", value=50)
    ],
    outputs="text",
    title="F1 Race Winner Predictor",
    description="Predict if a driver will win based on race conditions."
)

if __name__ == "__main__":
    iface.launch()
