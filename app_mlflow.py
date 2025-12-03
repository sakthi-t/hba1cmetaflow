import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__, template_folder="templates")

# Load patient data
df_patients = pd.read_csv("Data/dim_patients_final_rev01.csv")
patient_ids = df_patients['patient_id'].tolist()

# MLflow configuration
REPO_OWNER = "sakthi-t"
REPO_NAME = "hba1cmetaflow"
TRACKING_URI = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"

# Model state
loaded_model = None
model_run_id = "fb8d1d9483ef436baee58405144978e4"  # Known run with model
model_loaded = False

def setup_mlflow():
    """Configure MLflow with DagsHub credentials"""
    import mlflow
    import mlflow.sklearn

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

    return mlflow

def load_model_from_mlflow(run_id=None):
    """Load model directly from MLflow without downloading"""
    global loaded_model, model_run_id, model_loaded

    if model_loaded:
        return loaded_model

    if not run_id:
        run_id = model_run_id

    try:
        print(f"Loading model from MLflow...")
        print(f"Run ID: {run_id}")
        print(f"Tracking URI: {TRACKING_URI}")

        # Setup MLflow
        mlflow = setup_mlflow()

        # Method 1: Try using artifact_path with pickle model
        model_uri = f"runs:/{run_id}/model/model.pkl"
        print(f"Trying to load from: {model_uri}")

        try:
            # This will load the pickle file directly
            import mlflow.pyfunc
            model = mlflow.pyfunc.load_model(model_uri)
            print("✓ Model loaded as PyFunc!")

            # Extract the underlying sklearn model
            if hasattr(model, 'predict'):
                loaded_model = model
            else:
                # Try to get the underlying model
                loaded_model = model.unwrap_python_model()

            model_run_id = run_id
            model_loaded = True
            return loaded_model

        except Exception as e:
            print(f"PyFunc loading failed: {e}")

            # Method 2: Try using the directory path
            model_uri = f"runs:/{run_id}/model"
            print(f"Trying to load from: {model_uri}")

            # Load with a custom model loader that can handle pickle files
            import mlflow.models
            model_info = mlflow.models.get_model_info(model_uri)
            print(f"Model info: {model_info}")

            # Load the model using the local path trick
            # This uses MLflow's internal loading mechanism
            import mlflow.artifacts

            # Get the artifact repository
            from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
            repo = RunsArtifactRepository(TRACKING_URI)

            # Get the local path to the model
            local_path = repo.download_artifacts(artifact_path="model/model.pkl", run_id=run_id)
            print(f"Local path: {local_path}")

            # Load the pickle file
            import pickle
            with open(local_path, 'rb') as f:
                loaded_model = pickle.load(f)

            model_run_id = run_id
            model_loaded = True
            print("✓ Model loaded from local path!")
            return loaded_model

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_hba1c(patient_id, visited_date, sugar):
    """Make HBA1C prediction"""
    model = load_model_from_mlflow()
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
                         model_loaded=model_loaded,
                         model_run_id=model_run_id)

@app.route('/model-info')
def model_info():
    """API endpoint for model information"""
    return {
        "model_loaded": model_loaded,
        "model_run_id": model_run_id,
        "tracking_uri": TRACKING_URI,
        "model_url": f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow/#/experiments/0/runs/{model_run_id}"
    }

@app.route('/test-model')
def test_model():
    """Test the model without loading it into the main app"""
    model = load_model_from_mlflow()
    if model:
        import numpy as np
        test_input = np.array([[1, 150, 2023, 12, 3]])
        try:
            prediction = model.predict(test_input)
            return {
                "status": "success",
                "prediction": float(prediction[0]),
                "model_type": str(type(model))
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    else:
        return {
            "status": "error",
            "message": "Model not loaded"
        }

if __name__ == "__main__":
    print("=" * 60)
    print("HBA1C Prediction App - Direct MLflow Loading")
    print("=" * 60)
    print(f"Tracking URI: {TRACKING_URI}")
    print(f"Model Run ID: {model_run_id}")
    print("=" * 60)

    # Test model loading at startup (non-blocking)
    import threading
    load_thread = threading.Thread(target=load_model_from_mlflow)
    load_thread.daemon = True
    load_thread.start()

    print("\nStarting Flask app...")
    print("Access the app at: http://localhost:5000")
    print("Model will be loaded on first prediction request")
    app.run(host='0.0.0.0', port=5000, debug=True)