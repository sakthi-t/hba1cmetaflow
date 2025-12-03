import os
from dotenv import load_dotenv
import dagshub
import mlflow

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

print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Username: {os.getenv('MLFLOW_TRACKING_USERNAME')}")
print(f"Password: {'SET' if os.getenv('MLFLOW_TRACKING_PASSWORD') else 'NOT SET'}")

# List experiments
try:
    experiments = mlflow.search_experiments()
    print(f"\nFound {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp.name} (ID: {exp.experiment_id})")
except Exception as e:
    print(f"Error listing experiments: {e}")

# List registered models
try:
    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models()
    print(f"\nFound {len(models)} registered models:")
    for model in models:
        print(f"  - {model.name}")
        for version in model.latest_versions:
            print(f"    Version: {version.version}, Stage: {version.current_stage}")
except Exception as e:
    print(f"Error listing registered models: {e}")