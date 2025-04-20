from flask import Flask, request, jsonify
import torch
import pandas as pd
import mlflow.pytorch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from prometheus_flask_exporter import PrometheusMetrics
import joblib
# Flask app initialization
app = Flask(__name__)

#for prometheus
metrics = PrometheusMetrics(app)
# Load the trained model and scaler
model_uri = "mlruns/878626530893598251/fa4fbbdc580c48e696572d3607f7b9ab/artifacts/best_time_series_model"  
model = mlflow.pytorch.load_model(model_uri)

# Load the scaler (from your MLflow artifacts or locally saved file)
scaler = MinMaxScaler()
scaler_path = "scaler.pkl"  # Replace with your scaler file path
scaler = joblib.load(scaler_path)

# Configuration for sequence length
SEQUENCE_LENGTH = 10

# Helper function: Preprocess input data
def preprocess_data(input_data):
    """
    Scale the input data using the saved scaler and reshape for the model.
    :param input_data: List of numerical feature values.
    :return: Scaled and reshaped data ready for the model.
    """
    # Ensure input data is a numpy array
    input_array = np.array(input_data).reshape(-1, len(input_data[0]))
    scaled_data = scaler.transform(input_array)
    
    # Convert scaled data to sequences
    if len(scaled_data) < SEQUENCE_LENGTH:
        raise ValueError(f"Input data must have at least {SEQUENCE_LENGTH} rows.")
    
    sequences = []
    for i in range(len(scaled_data) - SEQUENCE_LENGTH + 1):
        sequences.append(scaled_data[i:i + SEQUENCE_LENGTH, :])
    
    return torch.tensor(sequences, dtype=torch.float32)

# Route: Health check
@app.route("/", methods=["GET"])
def health_check():
    return "Model is running!"

# Route: Predict future air quality
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract JSON data from request
        json_data = request.get_json()
        input_data = json_data.get("data")
        
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Preprocess data
        processed_data = preprocess_data(input_data)
        
        # Predict using the model
        model.eval()
        with torch.no_grad():
            predictions = model(processed_data)
        
        # Convert predictions to a list
        predictions = predictions.squeeze().tolist()
        return jsonify({"predictions": predictions})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
