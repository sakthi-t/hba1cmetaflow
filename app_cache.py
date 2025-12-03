import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template
import numpy as np
import pickle
import tempfile
import atexit

app = Flask(__name__, template_folder="templates")

# Load patient data
df_patients = pd.read_csv("Data/dim_patients_final_rev01.csv")
patient_ids = df_patients['patient_id'].tolist()

# Configuration
REPO_OWNER = "sakthi-t"
REPO_NAME = "hba1cmetaflow"
TRACKING_URI = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
LATEST_RUN_ID = "fb8d1d9483ef436baee58405144978e4"

# Model state (will be cached in memory after first load)
model_cache = {}
temp_dir = None

def cleanup_temp_dir():
    """Clean up temporary directory on exit"""
    global temp_dir
    if temp_dir and os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temp directory: {temp_dir}")

# Register cleanup
atexit.register(cleanup_temp_dir)

def get_mlflow_client():
    """Get configured MLflow client"""
    import mlflow
    import mlflow.tracking

    # Load environment variables
    load_dotenv()
    dagshub_token = os.getenv('DAGSHUB_TOKEN')

    if not dagshub_token:
        raise ValueError("DAGSHUB_TOKEN not found in .env file")

    # Set up authentication
    os.environ['MLFLOW_TRACKING_USERNAME'] = REPO_OWNER
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

    # Set tracking URI
    mlflow.set_tracking_uri(TRACKING_URI)

    return mlflow.tracking.MlflowClient()

def load_model_from_dagshub(run_id=None):
    """
    Load model from DagsHub - cached in memory after first load
    The model is downloaded once and kept in memory for subsequent requests
    """
    global model_cache, temp_dir

    # Check cache first
    if not run_id:
        run_id = LATEST_RUN_ID

    cache_key = f"model_{run_id}"
    if cache_key in model_cache:
        print(f"Using cached model for run {run_id}")
        return model_cache[cache_key]

    try:
        print(f"Loading model from DagsHub (first time for run {run_id})...")
        print(f"This will download the model once and cache it in memory")

        # Get MLflow client
        client = get_mlflow_client()

        # Create temporary directory if not exists
        global temp_dir
        if not temp_dir:
            temp_dir = tempfile.mkdtemp(prefix="hba1c_model_")
            print(f"Using temp directory: {temp_dir}")

        # Download model artifacts to temp directory
        print("Downloading model artifacts from DagsHub...")
        client.download_artifacts(run_id, "model", temp_dir)

        # Path to the model file
        model_path = os.path.join(temp_dir, "model.pkl")

        if not os.path.exists(model_path):
            # Check subdirectories
            for root, dirs, files in os.walk(temp_dir):
                if "model.pkl" in files:
                    model_path = os.path.join(root, "model.pkl")
                    break

        if not os.path.exists(model_path):
            raise FileNotFoundError("model.pkl not found in downloaded artifacts")

        # Load the model
        print(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Cache in memory
        model_cache[cache_key] = model

        print(f"✓ Model loaded and cached! Type: {type(model).__name__}")

        # Test prediction
        test_input = np.array([[1, 150, 2023, 12, 3]])
        prediction = model.predict(test_input)
        print(f"✓ Test prediction: {prediction[0]:.2f}")

        # Clean up temp files (but keep model in memory)
        # We keep the temp dir for now as the model might reference it
        # It will be cleaned up on app exit

        return model

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_hba1c(patient_id, visited_date, sugar):
    """Make HBA1C prediction using cached model"""
    model = load_model_from_dagshub()
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
                error = "Failed to make prediction. Model might not be available."

        except ValueError as e:
            error = "Invalid input. Please enter valid values."
        except Exception as e:
            error = f"An error occurred: {str(e)}"

    return render_template('index.html',
                         patient_ids=patient_ids,
                         prediction=prediction,
                         error=error,
                         model_loaded=bool(model_cache),
                         model_info={
                             'run_id': LATEST_RUN_ID,
                             'dagshub_url': f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow/#/experiments/0/runs/{LATEST_RUN_ID}",
                             'cache_size': len(model_cache)
                         })

@app.route('/model-info')
def model_info():
    """API endpoint for model information"""
    return {
        "model_loaded": bool(model_cache),
        "run_id": LATEST_RUN_ID,
        "tracking_uri": TRACKING_URI,
        "dagshub_url": f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow/#/experiments/0/runs/{LATEST_RUN_ID}",
        "cache_size": len(model_cache),
        "temp_dir": temp_dir
    }

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear model cache (useful for testing)"""
    global model_cache
    old_size = len(model_cache)
    model_cache.clear()
    return {
        "status": "success",
        "message": f"Cleared {old_size} cached models"
    }

if __name__ == "__main__":
    print("=" * 60)
    print("HBA1C Prediction App - Cached Model Loading")
    print("=" * 60)
    print(f"Repository: {REPO_OWNER}/{REPO_NAME}")
    print(f"Run ID: {LATEST_RUN_ID}")
    print(f"Model will be downloaded once and cached in memory")
    print(f"Access the app at: http://localhost:5000")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=True)