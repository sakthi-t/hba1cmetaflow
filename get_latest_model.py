import os
from dotenv import load_dotenv
import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd

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
print("GETTING LATEST MODEL FROM DAGSHUB")
print("=" * 60)

# Get the most recent run with metrics
runs = mlflow.search_runs(
    experiment_ids=["0"],
    filter_string="metrics.rmse_rf IS NOT NULL",  # Filter for runs with our model metrics
    order_by=["start_time DESC"],
    max_results=1
)

if len(runs) > 0:
    latest_run = runs.iloc[0]
    run_id = latest_run['run_id']

    print(f"Latest run with model: {run_id}")
    print(f"Metrics:")
    print(f"  - RMSE: {latest_run.get('metrics.rmse_rf', 'N/A')}")
    print(f"  - MAE: {latest_run.get('metrics.mae_rf', 'N/A')}")
    print(f"  - R2: {latest_run.get('metrics.r2_rf', 'N/A')}")

    # Construct model URI
    model_uri = f"runs:/{run_id}/model"

    print(f"\nModel URI for your app.py:")
    print(f"  model_uri = '{model_uri}'")

    # Test loading the model
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"\n✓ Model loaded successfully!")
        print(f"  Type: {type(model).__name__}")
        print(f"  Estimators: {getattr(model, 'n_estimators', 'N/A')}")

        # Test prediction
        import numpy as np
        # Format: patient_id, sugar, year, month, day
        test_data = np.array([[1, 150, 2023, 12, 3]])
        prediction = model.predict(test_data)
        print(f"\n  Test prediction (patient_id=1, sugar=150): {prediction[0]:.2f}")

    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")

else:
    print("No runs with model metrics found!")

print("\n" + "=" * 60)
print("LINKS")
print("=" * 60)
print(f"MLflow UI: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow")
print(f"Latest Run: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow/#/experiments/0/runs/{run_id if 'run_id' in locals() else 'N/A'}")
print("\nFor your app.py, use the model_uri shown above")
print("=" * 60)