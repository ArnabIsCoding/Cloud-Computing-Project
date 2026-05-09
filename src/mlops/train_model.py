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
    def __init__(self, tracking_uri: str = "http://localhost:2000"):
        # WE ARE BACK ONLINE! Pointing specifically to port 2000.
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Flight_Delay_Prediction")
        self.label_encoders = {}

    def load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        logger.info(f"Loading dataset from {data_path}")
        df = pd.read_parquet(data_path)

        logger.info("Applying feature engineering...")
        df = df[df['Cancelled'] == 0].copy()

        df['IsDelayed'] = (df['ArrDelay'] > 15).astype(int)
        df['DepHour'] = (df['CRSDepTime'] // 100).astype(int)

        features = ['Month', 'DayOfWeek', 'DepHour', 'UniqueCarrier', 'Origin', 'Dest']
        X = df[features].copy()
        y = df['IsDelayed']

        categorical_cols = ['UniqueCarrier', 'Origin', 'Dest']
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        return X, y

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
            mlflow.log_params(params)

            logger.info("Training XGBoost Classifier...")
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba)
            }

            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model, "xgboost-flight-delay-model")

            logger.info(f"Model Training Complete. Metrics: {metrics}")
            print("\n View your model in the MLflow UI at http://localhost:2000")

if __name__ == "__main__":
    INPUT_DATA = "data/processed/cleaned_flights.parquet"

    # Passing the port 2000 URI into the class
    MLFLOW_URI = "http://localhost:2000"
    pipeline = FlightDelayPredictor(tracking_uri=MLFLOW_URI)

    try:
        X, y = pipeline.load_and_preprocess_data(INPUT_DATA)
        pipeline.train_and_evaluate(X, y)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
