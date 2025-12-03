#!/usr/bin/env python3
"""
Script to update app.py with the latest model run ID from DagsHub
"""

import os
from dotenv import load_dotenv
import dagshub
import mlflow
import re

# Load environment variables
load_dotenv()

# Set up DagsHub
dagshub_token = os.getenv('DAGSHUB_TOKEN')
os.environ['MLFLOW_TRACKING_USERNAME'] = 'sakthi-t'
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
mlflow.set_tracking_uri("https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow")

# Define the latest run ID (from the most recent successful training)
LATEST_RUN_ID = "6c03e9695c26400b8076856ef73d3d83"

def update_app_with_new_run_id(run_id):
    """Update app.py with the new run ID"""
    app_file = "app.py"

    # Read the current app.py
    with open(app_file, 'r') as f:
        content = f.read()

    # Update the LATEST_RUN_ID line
    pattern = r'LATEST_RUN_ID = ".*?"'
    new_content = re.sub(pattern, f'LATEST_RUN_ID = "{run_id}"', content)

    # Write back to app.py
    with open(app_file, 'w') as f:
        f.write(new_content)

    print(f"✅ Updated app.py with run ID: {run_id}")

def print_current_model_info():
    """Print information about the current model"""
    print("\n" + "="*60)
    print("CURRENT MODEL INFORMATION")
    print("="*60)
    print(f"Latest Run ID: {LATEST_RUN_ID}")
    print(f"Model URI: runs:/{LATEST_RUN_ID}/model")
    print(f"MLflow UI: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow")
    print(f"Direct Run Link: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow/#/experiments/0/runs/{LATEST_RUN_ID}")
    print("="*60)

if __name__ == "__main__":
    print("\n🔄 HBA1C App Model Updater")
    print("\nCurrent configuration:")
    print_current_model_info()

    # For now, keep the current run ID
    # After running a new training, you can uncomment the following line:
    # update_app_with_new_run_id("<NEW_RUN_ID_HERE>")

    print("\nTo update with a new run ID after training:")
    print("1. Run: python3 mlops_mlflow_metaflow.py run")
    print("2. Copy the Run ID from the output")
    print("3. Update LATEST_RUN_ID in app.py manually or run this script with the new ID")
    print("\nOr run: python3 -c 'from update_app_model import update_app_with_new_run_id; update_app_with_new_run_id(\"<NEW_ID>\")'")