import logging
import sys
import warnings
from urllib.parse  import urlparse
import os
from dotenv import load_dotenv
import dagshub
import mlflow
import mlflow.sklearn

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from metaflow import FlowSpec, step, Parameter, current
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# from evidently import Report
# from evidently.presets import DataDriftPreset, RegressionPreset

class RegressionFlow(FlowSpec):

    data_path = Parameter('data_path', default="Data/fact_visits_final_rev01.csv")

    @step
    def start(self):
        logger.info("=== Starting Regression Flow ===")
        # initialize dagshub
        logger.info("Initializing DagsHub...")
        dagshub.init(repo_owner='sakthi-t', repo_name='hba1cmetaflow', mlflow=True)

        # Set MLflow authentication for DagsHub
        dagshub_token = os.getenv('DAGSHUB_TOKEN')
        if dagshub_token:
            os.environ['MLFLOW_TRACKING_USERNAME'] = 'sakthi-t'
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
            logger.info("MLflow authentication set from environment variables")
        else:
            logger.error("DAGSHUB_TOKEN not found in environment variables")
            raise ValueError("Please set DAGSHUB_TOKEN in your .env file")

        # set the MLFLOW tracking URI
        tracking_uri = "https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow"
        logger.info(f"Setting MLflow tracking URI to: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)

        self.remote_server_uri = tracking_uri
        self.tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        logger.info(f"Tracking URI scheme: {self.tracking_url_type_store}")

        logger.info("Starting MLflow run...")
        self.run = mlflow.start_run()
        logger.info(f"MLflow run started with ID: {self.run.info.run_id}")
        self.next(self.load_data)

    @step
    def load_data(self):
        logger.info(f"Loading data from: {self.data_path}")
        self.df_visits = pd.read_csv(self.data_path).copy()
        logger.info(f"Data loaded with shape: {self.df_visits.shape}")
        mlflow.log_param("data_path", self.data_path)
        self.next(self.transform_data)
    

    @step
    def transform_data(self):
        self.data = self.df_visits[['patient_id', 'visited_date', 'sugar', 'hba1c']]

        # Converting visited_date to datetime
        self.data['visited_date'] = pd.to_datetime(self.data['visited_date'])

        # Extracting year, month, and day from visited_date
        self.data['year'] = self.data['visited_date'].dt.year
        self.data['month'] = self.data['visited_date'].dt.month
        self.data['day'] = self.data['visited_date'].dt.day

        # Dropping the original visited date column
        self.data = self.data.drop(columns=['visited_date'])


        # Split data into reference and current datasets
        self.reference_data = self.data.sample(n=1000, replace=False)
        self.current_data = self.data.drop(self.reference_data.index)
        
        self.next(self.define_features_target)

    @step
    def define_features_target(self):
        self.X = self.data.drop(columns=['hba1c'])
        self.y = self.data['hba1c']
        self.next(self.train_test_split)

    @step
    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=42)
        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)
        self.next(self.randomforest_param)


    @step
    def randomforest_param(self):
        self.n_estimators = 100
        mlflow.log_param("num of estimators", self.n_estimators)
        self.next(self.fit_randomforest)


    @step
    def fit_randomforest(self):
        # Train your RandomForest model
        logger.info("Training RandomForest model...")
        self.X_train = np.array(self.X_train, copy=True) # code will fail without this line
        self.y_train = np.array(self.y_train, copy=True) # code will fail without this line
        self.model_rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
        self.model_rf.fit(self.X_train, self.y_train)
        logger.info("RandomForest model training completed")
        self.next(self.metrics_randomforest)

    @step
    def metrics_randomforest(self):
        logger.info("Calculating model metrics...")
        self.y_pred_rf = self.model_rf.predict(self.X_test)
        self.rmse_rf = sqrt(mean_squared_error(self.y_test, self.y_pred_rf))
        self.mae_rf = mean_absolute_error(self.y_test, self.y_pred_rf)
        self.r2_rf = r2_score(self.y_test, self.y_pred_rf)

        logger.info(f"Metrics - RMSE: {self.rmse_rf:.4f}, MAE: {self.mae_rf:.4f}, R2: {self.r2_rf:.4f}")

        logger.info("Logging model to MLflow...")
        try:
            # Log the model with the model name that exists in DagsHub
            logged_model = mlflow.sklearn.log_model(
                sk_model=self.model_rf,
                artifact_path="model",
                registered_model_name="hba1c",  # Use the model that exists in DagsHub
                input_example=self.X_test[:5],
                signature=mlflow.models.infer_signature(self.X_test, self.y_test[:5]),
                tags={'framework': 'sklearn', 'model_type': 'RandomForestRegressor', 'run_id': self.run.info.run_id}
            )
            logger.info(f"Model successfully logged and registered to 'hba1c'!")
            logger.info(f"Model artifact URI: {logged_model.model_uri}")

            # Save model info for app.py
            self.model_name = "hba1c"
            self.model_uri = f"models:/hba1c/latest"  # Use latest version

        except Exception as e:
            logger.warning(f"Could not register model using log_model: {str(e)}")
            logger.info("Falling back to simple model logging...")

            # Fallback: just log the model without registration
            logged_model = mlflow.sklearn.log_model(
                sk_model=self.model_rf,
                artifact_path="model",
                input_example=self.X_test[:5],
                signature=mlflow.models.infer_signature(self.X_test, self.y_test[:5])
            )
            logger.info(f"Model successfully logged to run artifacts!")
            logger.info(f"Model artifact URI: {logged_model.model_uri}")

            # Save run-based URI for app.py
            self.model_name = "hba1c"
            self.model_uri = f"runs:/{self.run.info.run_id}/model"

        mlflow.log_metric("rmse_rf", self.rmse_rf)
        mlflow.log_metric("mae_rf", self.mae_rf)
        mlflow.log_metric("r2_rf", self.r2_rf)
        logger.info("Metrics logged to MLflow")

        self.next(self.generate_reports)

    @step
    def generate_reports(self):

        # Create simple HTML reports with model performance
        html_report = f"""
        <html>
        <head><title>Model Performance Report</title></head>
        <body>
            <h1>Random Forest Model Performance</h1>
            <h2>Metrics:</h2>
            <ul>
                <li>RMSE: {self.rmse_rf:.4f}</li>
                <li>MAE: {self.mae_rf:.4f}</li>
                <li>R2 Score: {self.r2_rf:.4f}</li>
            </ul>

            <h2>Model Details:</h2>
            <ul>
                <li>Number of Estimators: {self.n_estimators}</li>
                <li>Training Data Size: {len(self.X_train)}</li>
                <li>Test Data Size: {len(self.X_test)}</li>
            </ul>

            <h2>Data Info:</h2>
            <ul>
                <li>Total Records: {len(self.data)}</li>
                <li>Features: patient_id, sugar, year, month, day</li>
                <li>Target: hba1c</li>
            </ul>
        </body>
        </html>
        """

        # Save the HTML report
        with open("model_performance_report.html", "w") as f:
            f.write(html_report)

        # Log report to MLflow
        mlflow.log_artifact("model_performance_report.html")

        self.next(self.end)
        
    
    @step
    def end(self):

        logger.info(f"Final Results - RandomForest - RMSE: {self.rmse_rf}, MAE: {self.mae_rf}, R2: {self.r2_rf}")

        # Log model URI for use in app.py
        if hasattr(self, 'model_uri'):
            logger.info(f"Model can be loaded from: {self.model_uri}")
            if hasattr(self, 'model_version') and self.model_version:
                logger.info(f"Model Registry: {self.model_name} version {self.model_version}")

        logger.info("Ending MLflow run...")
        mlflow.end_run()
        logger.info("=== Flow completed successfully ===")

        print("\n" + "="*60)
        print("MODEL DEPLOYMENT INFORMATION")
        print("="*60)
        print(f"Run ID: {self.run.info.run_id}")
        if hasattr(self, 'model_uri'):
            print(f"\nModel URI for app.py:")
            print(f"  model_uri = '{self.model_uri}'")
            if 'models:' in self.model_uri:
                print(f"  ✅ Model registered in DagsHub Model Registry!")
            else:
                print(f"  ⚠️  Using run-based URI (Model Registry might not be available)")
        print(f"\nMLflow UI:")
        print(f"  https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow")
        print(f"\nModels Registry:")
        print(f"  https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow/#/models")
        print(f"\nDirect Run Link:")
        print(f"  https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow/#/experiments/0/runs/{self.run.info.run_id}")
        print("="*60)

if __name__ == "__main__":
    RegressionFlow()
