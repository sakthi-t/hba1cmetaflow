import os
from dotenv import load_dotenv
import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

# Initialize DagsHub
dagshub.init(repo_owner='sakthi-t', repo_name='hba1cmetaflow', mlflow=True)

# Set authentication
dagshub_token = os.getenv('DAGSHUB_TOKEN')
os.environ['MLFLOW_TRACKING_USERNAME'] = 'sakthi-t'
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

# Set tracking URI
tracking_uri = "https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow"
mlflow.set_tracking_uri(tracking_uri)

print("=" * 60)
print("ACCESSING LATEST MODEL FROM DAGSHUB")
print("=" * 60)

# Get the latest run with model
try:
    # Search for runs with logged models
    runs = mlflow.search_runs(
        experiment_ids=["0"],
        filter_string="tags.mlflow.log-model.history",
        order_by=["start_time DESC"],
        max_results=1
    )

    if len(runs) > 0:
        latest_run_id = runs.iloc[0]['run_id']
        print(f"Latest run with model: {latest_run_id}")
        print(f"Run link: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow/#/experiments/0/runs/{latest_run_id}")

        # Load model from this run
        model_uri = f"runs:/{latest_run_id}/model"
        print(f"\nLoading model from: {model_uri}")

        model = mlflow.sklearn.load_model(model_uri)
        print(f"✓ Model loaded successfully!")
        print(f"Model type: {type(model)}")

        # Test model with sample data
        sample_input = np.array([[1, 150, 2023, 12, 3]])  # patient_id, sugar, year, month, day
        prediction = model.predict(sample_input)
        print(f"\nSample prediction: {prediction[0]:.2f}")

        # Show model details
        if hasattr(model, 'n_estimators'):
            print(f"\nModel details:")
            print(f"  - Estimators: {model.n_estimators}")
            print(f"  - Max depth: {model.max_depth}")
            print(f"  - Random state: {model.random_state}")

    else:
        print("No runs with models found!")

        # Try to get the latest run directly
        all_runs = mlflow.search_runs(experiment_ids=["0"], order_by=["start_time DESC"], max_results=1)
        if len(all_runs) > 0:
            latest_run_id = all_runs.iloc[0]['run_id']
            print(f"\nTrying latest run: {latest_run_id}")
            try:
                model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/model")
                print("✓ Successfully loaded model from latest run!")
            except Exception as e:
                print(f"✗ No model found in latest run: {e}")

except Exception as e:
    print(f"Error: {e}")

print()
print("=" * 60)
print("Model URLs:")
print(f"- MLflow UI: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow")
print(f"- Latest runs: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow/#/experiments/0")
print("=" * 60)