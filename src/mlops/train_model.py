import os
import logging
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FlightDelayPredictor:
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        # Connect to the MLflow server running in our Docker container
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Flight_Delay_Prediction")

        self.label_encoders = {}

    def load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        """Loads data and creates features for the ML model."""
        logger.info(f"Loading dataset from {data_path}")
        # In a real scenario, this loads the output from our batch pipeline
        df = pd.read_parquet(data_path)

        # 1. Feature Engineering
        logger.info("Applying feature engineering...")
        df = df[df['Cancelled'] == 0].copy()  # Only consider flights that actually flew

        # Create Binary Target: 1 if delayed > 15 mins, 0 otherwise
        df['IsDelayed'] = (df['ArrDelay'] > 15).astype(int)

        # Extract Hour from CRSDepTime (Scheduled Departure Time e.g., 1430 -> 14)
        df['DepHour'] = (df['CRSDepTime'] // 100).astype(int)

        # 2. Select Features
        features = ['Month', 'DayOfWeek', 'DepHour', 'UniqueCarrier', 'Origin', 'Dest']
        X = df[features].copy()
        y = df['IsDelayed']

        # 3. Encode Categorical Variables (Airlines and Airports)
        categorical_cols = ['UniqueCarrier', 'Origin', 'Dest']
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        return X, y

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series):
        """Trains the XGBoost model and logs metrics to MLflow."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define Hyperparameters
        params = {
            "objective": "binary:logistic",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "eval_metric": "auc",
            "seed": 42
        }

        logger.info("Starting MLflow run...")
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(params)

            # Initialize and train model
            logger.info("Training XGBoost Classifier...")
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)

            # Make Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Calculate Metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba)
            }

            # Log Metrics and Model to MLflow
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model, "xgboost-flight-delay-model")

            logger.info(f"Model Training Complete. Metrics: {metrics}")
            print("\nView your model in the MLflow UI at http://localhost:5000")

if __name__ == "__main__":
    # Ensure this points to the parquet output of the batch cleaning step
    INPUT_DATA = "data/processed/cleaned_flights.parquet"

    # We use the MLflow URI specified in docker-compose.yml
    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    pipeline = FlightDelayPredictor(tracking_uri=MLFLOW_URI)

    try:
        # Uncomment when you have run the batch step to generate the parquet file
        # X, y = pipeline.load_and_preprocess_data(INPUT_DATA)
        # pipeline.train_and_evaluate(X, y)
        logger.info("MLOps pipeline ready. Run the batch step first to generate data.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
