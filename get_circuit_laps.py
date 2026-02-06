import pandas as pd
import joblib
import json

def get_circuit_laps():
    # Load model to get the circuit encoder
    try:
        model_data = joblib.load('out/models/f1_model_data.pkl')
        le_circuit = model_data['le_circuit']
    except Exception as e:
        print(f"Error loading model: {e}")
        return {}

    # Load data
    results = pd.read_csv('data/results.csv')
    races = pd.read_csv('data/races.csv')
    circuits = pd.read_csv('data/circuits.csv')

    # Merge to get circuit reference names and laps
    df = results.merge(races, on='raceId').merge(circuits, on='circuitId')
    
    # Get max laps per circuit reference
    circuit_laps = df.groupby('circuitRef')['laps'].max().to_dict()
    
    # Filter only for circuits we have in our encoder
    final_mapping = {name: int(laps) for name, laps in circuit_laps.items() if name in le_circuit.classes_}
    
    return final_mapping

if __name__ == "__main__":
    mapping = get_circuit_laps()
    print(json.dumps(mapping, indent=4))
