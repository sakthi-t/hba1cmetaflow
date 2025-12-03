import os
from dotenv import load_dotenv
import dagshub
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

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
print("VERIFYING HBA1C MODEL IN DAGSHUB")
print("=" * 60)

client = MlflowClient()

# 1. Check if model exists
print("\n1. Checking if 'hba1c' model exists...")
try:
    registered_model = client.get_registered_model("hba1c")
    print(f"✓ Model 'hba1c' found!")
    print(f"  Description: {registered_model.description or 'None'}")
    print(f"  Creation timestamp: {registered_model.creation_timestamp}")
except Exception as e:
    print(f"✗ Model 'hba1c' not found: {e}")

# 2. Check all models
print("\n2. Listing all registered models...")
try:
    models = client.search_registered_models()
    if models:
        print(f"Found {len(models)} registered model(s):")
        for model in models:
            print(f"  - {model.name}")
    else:
        print("No registered models found")
except Exception as e:
    print(f"Error listing models: {e}")

# 3. Get recent runs with models
print("\n3. Checking recent runs with model artifacts...")
try:
    # Get recent runs
    runs = mlflow.search_runs(
        experiment_ids=["0"],
        order_by=["start_time DESC"],
        max_results=5
    )

    print(f"Found {len(runs)} recent runs:")
    for idx, run in runs.iterrows():
        run_id = run['run_id']
        print(f"\n  Run {idx + 1}: {run_id[:8]}...")
        print(f"    Start time: {run['start_time']}")

        # Check if this run has model artifacts
        try:
            model_artifacts = mlflow.artifacts.list_artifacts(run_id=run_id, path="model")
            if model_artifacts:
                print(f"    ✓ Has model artifacts")
                print(f"    Model URI: runs:/{run_id}/model")

                # Try to load and test the model
                try:
                    test_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
                    print(f"    ✓ Model can be loaded (Type: {type(test_model).__name__})")
                except Exception as load_error:
                    print(f"    ✗ Cannot load model: {load_error}")
            else:
                print(f"    ✗ No model artifacts")
        except Exception as e:
            print(f"    Could not check artifacts: {e}")

except Exception as e:
    print(f"Error checking runs: {e}")

# 4. Test loading model from different URIs
print("\n4. Testing model loading...")
test_uris = [
    "models:/hba1c/latest",
    "models:/hba1c/1",
    "models:/hba1c/Production",
    "models:/hba1c/Staging"
]

for uri in test_uris:
    try:
        print(f"\n  Trying: {uri}")
        model = mlflow.sklearn.load_model(uri)
        print(f"    ✓ Successfully loaded! Type: {type(model).__name__}")

        # Test a quick prediction
        import numpy as np
        test_input = np.array([[1, 120, 2023, 12, 3]])  # patient_id, sugar, year, month, day
        try:
            pred = model.predict(test_input)
            print(f"    Sample prediction: {pred[0]:.2f}")
        except:
            print(f"    Could not make prediction")

    except Exception as e:
        print(f"    ✗ Failed: {str(e)[:80]}...")

print("\n" + "=" * 60)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 60)
print("""
If you see 'No registered models found' but the model artifacts exist:
1. The model is logged in the run artifacts but not in the Model Registry
2. Use the run-based URI: 'runs:/<run-id>/model'
3. Check the MLflow UI directly at the run level

To view your model:
- Go to: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow
- Click on the latest run
- Look for 'model' folder in the artifacts section

For your app.py, use:
- If model registry works: model_uri = 'models:/hba1c/latest'
- If not: model_uri = 'runs:/<run-id>/model'
""")