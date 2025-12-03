#!/usr/bin/env python3
"""
Quick test to check if we can fetch the model from DagsHub
"""

import os
import sys
from dotenv import load_dotenv

print("=" * 60)
print("TESTING MODEL FETCH FROM DAGSHUB")
print("=" * 60)

# Load environment variables
load_dotenv()
dagshub_token = os.getenv('DAGSHUB_TOKEN')

if not dagshub_token:
    print("✗ DAGSHUB_TOKEN not found in .env file!")
    print("Please add your DAGSHUB_TOKEN to the .env file")
    sys.exit(1)

print(f"✓ DAGSHUB_TOKEN found: {'*' * 10}{dagshub_token[-4:]}")

# Try to import and configure MLflow
try:
    import mlflow
    import mlflow.sklearn
    print("✓ MLflow imported successfully")
except ImportError as e:
    print(f"✗ Failed to import MLflow: {e}")
    sys.exit(1)

# Configure MLflow
os.environ['MLFLOW_TRACKING_USERNAME'] = 'sakthi-t'
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
mlflow.set_tracking_uri("https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow")
print("✓ MLflow tracking URI set")

# Test run ID
run_id = "6c03e9695c26400b8076856ef73d3d83"
model_uri = f"runs:/{run_id}/model"

print(f"\nAttempting to load model...")
print(f"Run ID: {run_id}")
print(f"Model URI: {model_uri}")
print(f"Direct link: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow/#/experiments/0/runs/{run_id}")

# Add timeout to prevent hanging
import signal

def timeout_handler(signum, frame):
    print("\n✗ Request timed out - likely network issue with DagsHub")
    print("\nPossible solutions:")
    print("1. Check your internet connection")
    print("2. Try running with a VPN")
    print("3. Check if DagsHub is accessible at: https://dagshub.com")
    print("4. The model might not exist at the specified run ID")
    sys.exit(1)

# Set 30 second timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)

try:
    # Attempt to load model
    model = mlflow.sklearn.load_model(model_uri)
    signal.alarm(0)  # Cancel timeout

    print("✅ SUCCESS! Model loaded from DagsHub!")
    print(f"Model type: {type(model).__name__}")

    # Test a prediction
    import numpy as np
    test_data = np.array([[1, 150, 2023, 12, 3]])
    try:
        prediction = model.predict(test_data)
        print(f"✓ Test prediction successful: {prediction[0]:.4f}")
        print("\n✅ Everything works! Your app.py should be able to load the model.")
    except Exception as pred_error:
        print(f"⚠️  Model loaded but prediction failed: {pred_error}")

except Exception as e:
    signal.alarm(0)  # Cancel timeout
    print(f"\n✗ Failed to load model: {str(e)}")
    print("\nThis could mean:")
    print("1. The run ID doesn't exist")
    print("2. The model wasn't properly logged to MLflow")
    print("3. Authentication issue with DagsHub")
    print("\nTry running a new training to get a fresh run ID:")
    print("  python3 mlops_mlflow_metaflow.py run")

print("\n" + "=" * 60)