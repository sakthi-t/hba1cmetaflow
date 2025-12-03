import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template
import numpy as np
import requests
import tempfile
import pickle
import shutil
import json

app = Flask(__name__, template_folder="templates")

# Load patient data
df_patients = pd.read_csv("Data/dim_patients_final_rev01.csv")
patient_ids = df_patients['patient_id'].tolist()

# Configuration
REPO_OWNER = "sakthi-t"
REPO_NAME = "hba1cmetaflow"
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')
MODEL_CACHE_DIR = "model_cache"

# Ensure cache directory exists
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Model state
loaded_model = None
model_run_id = None
model_loaded = False

def get_latest_run_with_model():
    """Get the latest run ID that has model artifacts"""
    try:
        # Configure MLflow
        import mlflow
        import mlflow.tracking

        os.environ['MLFLOW_TRACKING_USERNAME'] = REPO_OWNER
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
        mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")

        client = mlflow.tracking.MlflowClient()

        # Get all runs and find the latest one with model artifacts
        experiments = client.search_experiments()
        if experiments:
            runs = client.search_runs(experiment_ids=[experiments[0].experiment_id],
                                   order_by=["start_time DESC"],
                                   max_results=10)

            for run in runs:
                run_id = run.info.run_id
                # Check if this run has model.pkl by trying to list artifacts
                try:
                    # Use MLflow client to check for model artifacts
                    # Since DagsHub has API limitations, we'll try a different approach
                    return run_id
                except:
                    continue

        return None

    except Exception as e:
        print(f"Error getting latest run: {e}")
        return None

def download_model_from_dagshub(run_id):
    """Download model.pkl directly from DagsHub using API"""
    try:
        # DagsHub raw file URL pattern
        url = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}/raw/mlflow/{run_id}/artifacts/model/model.pkl"

        print(f"Downloading model from: {url}")

        headers = {
            "Authorization": f"Bearer {DAGSHUB_TOKEN}"
        }

        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        # Save to cache
        cache_path = os.path.join(MODEL_CACHE_DIR, f"model_{run_id}.pkl")
        with open(cache_path, 'wb') as f:
            f.write(response.content)

        print(f"✓ Model downloaded and cached to: {cache_path}")
        return cache_path

    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

def load_model_from_dagshub(run_id=None):
    """Load model from DagsHub - either from cache or download"""
    global loaded_model, model_run_id, model_loaded

    # Get latest run ID if not provided
    if not run_id:
        run_id = "fb8d1d9483ef436baee58405144978e4"  # Default to known good run

    # Check if we already have this model cached
    cache_path = os.path.join(MODEL_CACHE_DIR, f"model_{run_id}.pkl")
    if os.path.exists(cache_path) and run_id == model_run_id and model_loaded:
        print(f"Using cached model for run {run_id}")
        return loaded_model

    # Try to download if not cached
    downloaded_path = download_model_from_dagshub(run_id)
    if downloaded_path:
        try:
            with open(downloaded_path, 'rb') as f:
                loaded_model = pickle.load(f)
            model_run_id = run_id
            model_loaded = True
            print(f"✓ Model loaded successfully! Run ID: {run_id}")
            print(f"Model type: {type(loaded_model).__name__}")
            return loaded_model
        except Exception as e:
            print(f"Error loading downloaded model: {e}")

    return None

def get_model_from_mlflow_artifacts(run_id):
    """Alternative method using MLflow artifact download"""
    try:
        import mlflow
        import mlflow.tracking

        os.environ['MLFLOW_TRACKING_USERNAME'] = REPO_OWNER
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
        mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")

        client = mlflow.tracking.MlflowClient()

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        # Download artifacts
        client.download_artifacts(run_id, "model", temp_dir)

        # Find model.pkl
        model_path = None
        for root, dirs, files in os.walk(temp_dir):
            if "model.pkl" in files:
                model_path = os.path.join(root, "model.pkl")
                break

        if model_path:
            # Load the model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Save to cache
            cache_path = os.path.join(MODEL_CACHE_DIR, f"model_{run_id}.pkl")
            shutil.copy2(model_path, cache_path)

            # Clean up temp dir
            shutil.rmtree(temp_dir)

            return model

        shutil.rmtree(temp_dir)
        return None

    except Exception as e:
        print(f"Error with MLflow download: {e}")
        return None

def ensure_model_loaded(run_id=None):
    """Ensure model is loaded"""
    global loaded_model, model_loaded

    if model_loaded:
        return loaded_model

    # Try multiple methods
    print("Attempting to load model...")

    # Method 1: Direct download from DagsHub
    model = load_model_from_dagshub(run_id)
    if model:
        return model

    # Method 2: MLflow artifacts
    print("Trying MLflow artifact download...")
    model = get_model_from_mlflow_artifacts(run_id or "fb8d1d9483ef436baee58405144978e4")
    if model:
        global model_run_id
        model_run_id = run_id or "fb8d1d9483ef436baee58405144978e4"
        loaded_model = model
        model_loaded = True
        return model

    print("✗ All methods failed. Model not available.")
    return None

def predict_hba1c(patient_id, visited_date, sugar):
    """Make HBA1C prediction"""
    model = ensure_model_loaded()
    if model is None:
        return None

    try:
        # Prepare input data
        visited_date = pd.to_datetime(visited_date)
        input_data = np.array([[
            int(patient_id),
            float(sugar),
            visited_date.year,
            visited_date.month,
            visited_date.day
        ]])

        # Make prediction
        prediction = model.predict(input_data)
        return float(prediction[0])
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            patient_id = request.form['patient_id']
            visited_date = request.form['visited_date']
            sugar = float(request.form['sugar'])

            prediction = predict_hba1c(patient_id, visited_date, sugar)

            if prediction is None:
                error = "Failed to make prediction. Model might not be loaded."

        except ValueError as e:
            error = "Invalid input. Please enter valid values."
        except Exception as e:
            error = f"An error occurred: {str(e)}"

    return render_template('index.html',
                         patient_ids=patient_ids,
                         prediction=prediction,
                         error=error,
                         model_loaded=model_loaded,
                         model_run_id=model_run_id)

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """Endpoint to reload model from a specific run ID"""
    run_id = request.form.get('run_id')
    if run_id:
        global model_loaded
        model_loaded = False  # Force reload
        model = ensure_model_loaded(run_id)
        if model:
            return {"status": "success", "message": f"Model loaded from run {run_id}"}
        else:
            return {"status": "error", "message": "Failed to load model"}
    return {"status": "error", "message": "No run ID provided"}

@app.route('/model-info')
def model_info():
    """API endpoint for model information"""
    return {
        "model_loaded": model_loaded,
        "model_run_id": model_run_id,
        "model_type": type(loaded_model).__name__ if loaded_model else None,
        "dagshub_url": f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow/#/experiments/0/runs/{model_run_id}" if model_run_id else None
    }

if __name__ == "__main__":
    print("=" * 60)
    print("HBA1C Prediction App - Dynamic Model Loading")
    print("=" * 60)
    print(f"Repository: {REPO_OWNER}/{REPO_NAME}")
    print(f"Model cache directory: {MODEL_CACHE_DIR}")
    print("=" * 60)

    # Try to load model at startup
    ensure_model_loaded()

    print("\nStarting Flask app...")
    print("Access the app at: http://localhost:5000")
    print("Model will be loaded on first request if not already loaded")
    app.run(host='0.0.0.0', port=5000, debug=True)