#!/usr/bin/env python3
"""
Simple script to train and save model to DagsHub
"""

import os
from dotenv import load_dotenv
import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

print("=" * 60)
print("TRAINING AND SAVING MODEL TO DAGSHUB")
print("=" * 60)

# Initialize DagsHub
print("\n1. Initializing DagsHub...")
dagshub.init(repo_owner='sakthi-t', repo_name='hba1cmetaflow', mlflow=True)

# Set up MLflow
dagshub_token = os.getenv('DAGSHUB_TOKEN')
os.environ['MLFLOW_TRACKING_USERNAME'] = 'sakthi-t'
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
tracking_uri = "https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow"
mlflow.set_tracking_uri(tracking_uri)

print(f"✓ MLflow tracking URI: {tracking_uri}")

# Load data
print("\n2. Loading data...")
df = pd.read_csv("Data/fact_visits_final_rev01.csv").copy()
print(f"✓ Data loaded: {df.shape}")

# Prepare features
print("\n3. Preparing features...")
data = df[['patient_id', 'visited_date', 'sugar', 'hba1c']].copy()
data['visited_date'] = pd.to_datetime(data['visited_date'])
data['year'] = data['visited_date'].dt.year
data['month'] = data['visited_date'].dt.month
data['day'] = data['visited_date'].dt.day
data = data.drop(columns=['visited_date'])

X = data.drop(columns=['hba1c'])
y = data['hba1c']

print(f"✓ Features prepared: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"✓ Train size: {len(X_train)}, Test size: {len(X_test)}")

# Start MLflow run
print("\n4. Training model...")
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✓ Model trained!")

    # Calculate metrics
    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Metrics:")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - MAE: {mae:.4f}")
    print(f"  - R2: {r2:.4f}")

    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("model_type", "RandomForestRegressor")

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Save model with explicit logging
    print("\n5. Saving model...")

    # Create a temporary directory for the model
    import tempfile
    import shutil
    import joblib

    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "model.pkl")
    joblib.dump(model, model_path)

    # Log model artifact
    mlflow.log_artifact(model_path, "model")
    print("✓ Model saved as artifact!")

    # Clean up
    shutil.rmtree(temp_dir)

    # Try to log model in MLflow format
    try:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn_model",
            registered_model_name="hba1c"
        )
        print("✓ Model logged in MLflow format!")
    except Exception as e:
        print(f"⚠️  Could not log in MLflow format: {e}")

    print(f"\n✅ Training complete!")
    print(f"Run ID: {run_id}")
    print(f"MLflow UI: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow/#/experiments/0/runs/{run_id}")

    # Update app.py with new run ID
    app_file = "app.py"
    if os.path.exists(app_file):
        with open(app_file, 'r') as f:
            content = f.read()

        import re
        pattern = r'LATEST_RUN_ID = ".*?"'
        new_content = re.sub(pattern, f'LATEST_RUN_ID = "{run_id}"', content)

        with open(app_file, 'w') as f:
            f.write(new_content)

        print(f"✓ Updated {app_file} with new run ID")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"To use this model in your app:")
print(f"1. The app.py has been updated with the new run ID")
print(f"2. Run: python3 app.py")
print(f"3. The model will be loaded from runs:/{run_id}/model")
print("=" * 60)