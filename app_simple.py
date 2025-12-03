import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder="templates")

# Load patient data
df_patients = pd.read_csv("Data/dim_patients_final_rev01.csv")
patient_ids = df_patients['patient_id'].tolist()

# Model configuration
LATEST_RUN_ID = "6c03e9695c26400b8076856ef73d3d83"  # Latest successful run
model_uri = f"runs:/{LATEST_RUN_ID}/model"

# Global variables for model
loaded_model = None
model_loaded = False

def setup_mlflow():
    """Setup MLflow authentication"""
    # Set up DagsHub authentication
    dagshub_token = os.getenv('DAGSHUB_TOKEN')
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'sakthi-t'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

    # Set the MLflow tracking URI to DagsHub
    mlflow.set_tracking_uri("https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow")

def load_model_if_needed():
    """Load model only when needed"""
    global loaded_model, model_loaded
    if model_loaded:
        return loaded_model

    try:
        print(f"Attempting to load model from: {model_uri}")
        setup_mlflow()
        loaded_model = mlflow.sklearn.load_model(model_uri)
        model_loaded = True
        print(f"✓ Model loaded successfully!")
        return loaded_model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Please check if the run ID is correct and the model exists in DagsHub")
        model_loaded = False
        return None

def predict_hba1c(patient_id, visited_date, sugar):
    """Make HBA1C prediction"""
    model = load_model_if_needed()
    if model is None:
        return None

    try:
        # Prepare input data
        visited_date = pd.to_datetime(visited_date)
        data = {
            'patient_id': [patient_id],
            'sugar': [sugar],
            'year': [visited_date.year],
            'month': [visited_date.month],
            'day': [visited_date.day]
        }
        input_df = pd.DataFrame(data)

        # Make prediction
        prediction = model.predict(input_df)
        return prediction[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            patient_id = request.form['patient_id']
            visited_date = request.form['visited_date']
            sugar = float(request.form['sugar'])
            prediction = predict_hba1c(patient_id, visited_date, sugar)
        except Exception as e:
            print(f"Form processing error: {e}")

    # Check if model is loaded without forcing a load
    current_model_loaded = model_loaded

    return render_template('index.html',
                         patient_ids=patient_ids,
                         prediction=prediction,
                         model_loaded=current_model_loaded)

@app.route('/model-status')
def model_status():
    """API endpoint to check model status"""
    if not model_loaded:
        # Try loading
        load_model_if_needed()

    return {
        "model_loaded": model_loaded,
        "run_id": LATEST_RUN_ID,
        "model_uri": model_uri
    }

if __name__ == "__main__":
    print("=" * 60)
    print("HBA1C Prediction App")
    print("=" * 60)
    print(f"Model Run ID: {LATEST_RUN_ID}")
    print(f"Model URI: {model_uri}")
    print(f"MLflow UI: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow")
    print("=" * 60)
    print("\nStarting Flask app...")
    print("The model will be loaded on first prediction request.")
    app.run(host='0.0.0.0', port=5000, debug=True)