import os
from dotenv import load_dotenv
import dagshub
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor

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
print("TESTING HBA1C MODEL REGISTRATION")
print("=" * 60)
print(f"Tracking URI: {tracking_uri}")

# Check if model exists
try:
    client = mlflow.tracking.MlflowClient()

    # Check if model 'hba1c' exists
    try:
        model_info = client.get_registered_model("hba1c")
        print(f"\n✓ Found existing model 'hba1c'")
        print(f"  - Description: {model_info.description or 'None'}")
        print(f"  - Creation timestamp: {model_info.creation_timestamp}")

        # List existing versions
        if model_info.latest_versions:
            print(f"\n  Existing versions:")
            for version in model_info.latest_versions:
                print(f"    - Version {version.version}: Stage = {version.current_stage}")

    except Exception as e:
        print(f"\n✗ Model 'hba1c' not found or error: {e}")
        print("\nCreating model 'hba1c'...")
        try:
            client.create_registered_model(
                name="hba1c",
                description="HBA1C prediction model for healthcare analytics"
            )
            print("✓ Model 'hba1c' created successfully!")
        except Exception as create_error:
            print(f"✗ Failed to create model: {create_error}")

    # Test with a simple model
    print("\n" + "-" * 40)
    print("Testing model registration with sample data...")

    with mlflow.start_run() as run:
        # Create a simple model
        X_train = np.random.rand(100, 5)
        y_train = np.random.rand(100) * 10

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Log metrics
        accuracy = model.score(X_train, y_train)
        mlflow.log_metric("train_score", accuracy)

        # Register the model
        print(f"Logging model to 'hba1c' registry...")
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="hba1c",
            tags={"type": "test", "framework": "sklearn"}
        )

        print(f"✓ Model logged successfully!")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Model URI: {model_info.model_uri}")

        # Get the version
        if hasattr(model_info, 'version'):
            print(f"  Model Version: {model_info.version}")

except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("CHECKING MODEL ACCESS")
print("=" * 60)

# Test loading the model
try:
    # Try to load latest version
    model_uri = "models:/hba1c/latest"
    print(f"Loading model from: {model_uri}")
    loaded_model = mlflow.sklearn.load_model(model_uri)
    print("✓ Successfully loaded latest model!")

    # Test prediction
    test_input = np.random.rand(1, 5)
    prediction = loaded_model.predict(test_input)
    print(f"  Test prediction: {prediction[0]:.4f}")

except Exception as e:
    print(f"✗ Failed to load model: {e}")

    # Try loading from production stage
    try:
        model_uri = "models:/hba1c/Production"
        print(f"\nTrying Production stage: {model_uri}")
        loaded_model = mlflow.sklearn.load_model(model_uri)
        print("✓ Successfully loaded Production model!")
    except Exception as e2:
        print(f"✗ Failed to load Production model: {e2}")

print("\n" + "=" * 60)
print("LINKS")
print("=" * 60)
print(f"MLflow UI: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow")
print(f"Models: https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow/#/models")
print("=" * 60)