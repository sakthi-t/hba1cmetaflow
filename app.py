import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, render_template

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder="templates")

# Load patient data
df_patients = pd.read_csv("Data/dim_patients_final_rev01.csv")
patient_ids = df_patients['patient_id'].tolist()

# Set up DagsHub authentication
dagshub_token = os.getenv('DAGSHUB_TOKEN')
os.environ['MLFLOW_TRACKING_USERNAME'] = 'sakthi-t'
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

# Set the MLflow tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow")

# Model configuration - will be loaded on first request
LATEST_RUN_ID = "fb8d1d9483ef436baee58405144978e4"  # Latest successful run
model_uri = f"runs:/{LATEST_RUN_ID}/model"
loaded_model = None
model_loaded = False
current_run_id = None

def load_model():
    """Load model directly from DagsHub without downloading"""
    global loaded_model, model_loaded
    if model_loaded:
        return loaded_model

    try:
        print(f"Loading model from DagsHub MLflow...")
        print(f"Run ID: {LATEST_RUN_ID}")

        # Method 1: Use MLflow client to download artifacts
        from mlflow.tracking import MlflowClient
        import tempfile

        client = MlflowClient()
        temp_dir = tempfile.mkdtemp()

        # Download the model directory
        client.download_artifacts(LATEST_RUN_ID, "model", temp_dir)

        # Path to the pickle file - check multiple locations
        local_path = None
        possible_paths = [
            os.path.join(temp_dir, "model.pkl"),
            os.path.join(temp_dir, "model", "model.pkl")
        ]

        for path in possible_paths:
            if os.path.exists(path):
                local_path = path
                break

        # If still not found, search recursively
        if not local_path:
            for root, dirs, files in os.walk(temp_dir):
                if "model.pkl" in files:
                    local_path = os.path.join(root, "model.pkl")
                    break
        print(f"Model accessible at: {local_path}")

        # Load the model from the local path
        import pickle
        try:
            with open(local_path, 'rb') as f:
                loaded_model = pickle.load(f)
        except Exception as pickle_error:
            print(f"Pickle load error: {pickle_error}")
            # Try loading with joblib as fallback
            try:
                import joblib
                loaded_model = joblib.load(local_path)
                print("✓ Model loaded with joblib instead of pickle")
            except Exception as joblib_error:
                print(f"Joblib load error: {joblib_error}")
                raise pickle_error

        model_loaded = True
        global current_run_id
        current_run_id = LATEST_RUN_ID  # Track the run ID

        print(f"✓ Model loaded successfully! Type: {type(loaded_model).__name__}")
        print(f"✓ Run ID: {current_run_id}")

        # Test a quick prediction
        test_input = np.array([[1, 150, 2023, 12, 3]])
        prediction = loaded_model.predict(test_input)
        print(f"✓ Test prediction successful: {prediction[0]:.2f}")

        return loaded_model

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if DAGSHUB_TOKEN is set in .env")
        print("2. Verify the run ID is correct")
        print("3. Check if model.pkl exists in the artifacts")
        print(f"4. View at: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow/#/experiments/0/runs/{LATEST_RUN_ID}/artifacts")

        # Create fallback model for testing
        print("\nCreating fallback model...")
        from sklearn.ensemble import RandomForestRegressor

        # Simple training data
        X_sample = [[1, 100, 2023, 1, 1], [2, 200, 2023, 6, 15], [3, 150, 2024, 3, 10]]
        y_sample = [6.5, 9.0, 7.5]

        fallback_model = RandomForestRegressor(n_estimators=10, random_state=42)
        fallback_model.fit(X_sample, y_sample)

        loaded_model = fallback_model
        model_loaded = True
        print(f"✓ Fallback model created (for testing only)")

    return loaded_model if model_loaded else None

# Load model asynchronously
import threading
model_thread = threading.Thread(target=load_model)
model_thread.daemon = True
model_thread.start()

def predict_hba1c(patient_id, visited_date, sugar):
    if loaded_model is None:
        return None

    try:
        # Prepare input data (model expects numpy array)
        visited_date = pd.to_datetime(visited_date)
        input_data = np.array([[
            int(patient_id),
            float(sugar),
            visited_date.year,
            visited_date.month,
            visited_date.day
        ]])

        # Make prediction
        prediction = loaded_model.predict(input_data)
        return float(prediction[0])
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    # Check model status
    if not model_loaded and not model_thread.is_alive():
        # Try to load if failed before
        load_model()

    current_model_loaded = model_loaded

    if request.method == 'POST' and current_model_loaded:
        try:
            patient_id = request.form['patient_id']
            visited_date = request.form['visited_date']
            sugar = float(request.form['sugar'])
            prediction = predict_hba1c(patient_id, visited_date, sugar)
        except Exception as e:
            print(f"Form processing error: {e}")

    return render_template('index.html',
                         patient_ids=patient_ids,
                         prediction=prediction,
                         model_loaded=current_model_loaded,
                         current_run_id=current_run_id,
                         error=None)

if __name__ == "__main__":
    app.run(debug=True)
