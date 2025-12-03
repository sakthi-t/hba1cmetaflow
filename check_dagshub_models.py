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
print("DAGSHUB MLFLOW MODEL REGISTRY CHECK")
print("=" * 60)
print(f"Tracking URI: {tracking_uri}")
print(f"Repository: https://dagshub.com/sakthi-t/hba1cmetaflow")
print()

# Check experiments
try:
    experiments = mlflow.search_experiments()
    print(f"Found {len(experiments)} experiment(s):")
    for exp in experiments:
        print(f"  - {exp.name} (ID: {exp.experiment_id})")
        # List runs in this experiment
        runs = mlflow.search_runs(exp.experiment_ids)
        if len(runs) > 0:
            print(f"    Latest run: {runs.iloc[-1]['info.run_id'][:8]}...")
except Exception as e:
    print(f"Error listing experiments: {e}")

print()

# Check model registry using different approaches
client = MlflowClient()

print("Model Registry Status:")
print("-" * 30)

# Method 1: Try to get registered models
try:
    registered_models = client.search_registered_models()
    if registered_models:
        print(f"Found {len(registered_models)} registered model(s):")
        for model in registered_models:
            print(f"\n  Model: {model.name}")
            print(f"  Description: {model.description or 'No description'}")
            print(f"  Creation timestamp: {model.creation_timestamp}")

            # List versions
            if hasattr(model, 'latest_versions'):
                for version in model.latest_versions:
                    print(f"    Version: {version.version}")
                    print(f"    Stage: {version.current_stage}")
                    print(f"    Run ID: {version.run_id[:8]}...")
    else:
        print("No registered models found via search_registered_models()")
except Exception as e:
    print(f"Error with search_registered_models(): {e}")

print()

# Method 2: Try to get model versions directly
try:
    model_name = "random_forest_model"
    model_versions = client.search_model_versions(f"name='{model_name}'")
    if model_versions:
        print(f"Found {len(model_versions)} versions of '{model_name}':")
        for version in model_versions:
            print(f"  Version: {version.version}")
            print(f"  Stage: {version.current_stage}")
            print(f"  Run ID: {version.run_id}")
            print(f"  Creation time: {version.creation_timestamp}")
            print(f"  Source: {version.source}")
    else:
        print(f"No versions found for model '{model_name}'")
except Exception as e:
    print(f"Error with search_model_versions(): {e}")

print()

# Method 3: Direct API endpoint check
print("Direct Model Loading Test:")
print("-" * 30)
try:
    # Try to load the model using the registered model URI
    model_uri = "models:/random_forest_model/Production"
    print(f"Trying to load model from: {model_uri}")
    loaded_model = mlflow.sklearn.load_model(model_uri)
    print("✓ Successfully loaded model from Production stage!")
    print(f"Model type: {type(loaded_model)}")
except Exception as e:
    print(f"✗ Failed to load from Production: {e}")

    try:
        model_uri = "models:/random_forest_model/latest"
        print(f"\nTrying to load model from: {model_uri}")
        loaded_model = mlflow.sklearn.load_model(model_uri)
        print("✓ Successfully loaded model from latest version!")
        print(f"Model type: {type(loaded_model)}")
    except Exception as e2:
        print(f"✗ Failed to load from latest: {e2}")

print()
print("=" * 60)
print("To view models in DagsHub UI:")
print("1. Go to: https://dagshub.com/sakthi-t/hba1cmetaflow/mlflow")
print("2. Click on 'Models' tab in the MLflow UI")
print("3. Look for 'random_forest_model' in the registered models")
print("=" * 60)