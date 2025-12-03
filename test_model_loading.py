#!/usr/bin/env python3
"""
Test script to verify model loading from DagsHub
"""

import os
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
import numpy as np

print("=" * 60)
print("TESTING MODEL LOADING FROM DAGSHUB")
print("=" * 60)

# Load environment variables
load_dotenv()

# Setup MLflow
dagshub_token = os.getenv('DAGSHUB_TOKEN')
os.environ['MLFLOW_TRACKING_USERNAME'] = 'sakthi-t'
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
mlflow.set_tracking_uri("https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow")

# Test different run IDs
run_ids_to_test = [
    "6c03e9695c26400b8076856ef73d3d83",  # Latest from mlops_mlflow_metaflow.py
    "1ff0e6e042d7475698dccff93b5c35b0",  # From earlier run
    "457bff9f5b4f4611b8fb4e9f69ea8307",  # Another run
]

for run_id in run_ids_to_test:
    print(f"\nTrying run ID: {run_id}")
    model_uri = f"runs:/{run_id}/model"

    try:
        print(f"  Loading from: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print(f"  ✓ Model loaded successfully!")
        print(f"    Type: {type(model).__name__}")

        # Test prediction
        test_input = np.array([[1, 150, 2023, 12, 3]])  # patient_id, sugar, year, month, day
        prediction = model.predict(test_input)
        print(f"    Test prediction: {prediction[0]:.4f}")

        print(f"\n  ✅ This model works! Use this run ID in app.py")
        print(f"  Update LATEST_RUN_ID = '{run_id}' in app.py")
        break

    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:100]}...")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("If none of the run IDs worked, you may need to:")
print("1. Run a new training: python3 mlops_mlflow_metaflow.py run")
print("2. Copy the new Run ID from the output")
print("3. Update LATEST_RUN_ID in app.py")
print("=" * 60)