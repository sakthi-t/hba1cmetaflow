#!/usr/bin/env python3
"""
Debug MLflow model access
"""

import os
from dotenv import load_dotenv
import mlflow
import mlflow.tracking

# Load environment variables
load_dotenv()
dagshub_token = os.getenv('DAGSHUB_TOKEN')

print("=" * 60)
print("DEBUGGING MLFLOW MODEL ACCESS")
print("=" * 60)

# Configure MLflow
os.environ['MLFLOW_TRACKING_USERNAME'] = 'sakthi-t'
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
tracking_uri = "https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow"
mlflow.set_tracking_uri(tracking_uri)

print(f"Tracking URI: {tracking_uri}")
print(f"Username: sakthi-t")
print(f"Password: {'*' * 10}{dagshub_token[-4:] if dagshub_token else 'None'}")

# Test basic connection
print("\n1. Testing basic connection...")
try:
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    print(f"✓ Connected! Found {len(experiments)} experiment(s)")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    exit(1)

# Test getting runs
print("\n2. Searching for runs with models...")
try:
    # Get all runs
    runs = client.search_runs(experiment_ids=[experiments[0].experiment_id])
    print(f"Found {len(runs)} total runs")

    # Find runs with model artifacts
    model_runs = []
    for run in runs[:5]:  # Check last 5 runs
        run_id = run.info.run_id
        print(f"\nChecking run: {run_id}")
        print(f"  Status: {run.info.status}")
        print(f"  Start time: {run.info.start_time}")

        # Check if model artifacts exist
        artifacts = client.list_artifacts(run_id)
        model_artifacts = [a for a in artifacts if a.path == "model"]

        if model_artifacts:
            model_runs.append(run_id)
            print(f"  ✓ Has model artifacts!")

            # Try to load this model
            model_uri = f"runs:/{run_id}/model"
            try:
                model = mlflow.sklearn.load_model(model_uri)
                print(f"  ✓ Model loaded successfully!")

                # Test prediction
                import numpy as np
                test_input = np.array([[1, 150, 2023, 12, 3]])
                prediction = model.predict(test_input)
                print(f"  ✓ Test prediction: {prediction[0]:.2f}")

                print(f"\n✅ SUCCESS! Use this run ID: {run_id}")

                # Save working run ID
                with open("working_run_id.txt", "w") as f:
                    f.write(run_id)
                print(f"Saved to: working_run_id.txt")
                break

            except Exception as load_error:
                print(f"  ✗ Failed to load: {load_error}")
        else:
            print(f"  - No model artifacts")

    if not model_runs:
        print("\n⚠️  No runs with model artifacts found in recent runs")

except Exception as e:
    print(f"✗ Error searching runs: {e}")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)