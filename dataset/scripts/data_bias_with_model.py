import pickle
import numpy as np
import pandas as pd
import logging
from fairlearn.metrics import MetricFrame
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from typing import Dict
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelBiasDetector:
    def __init__(self, model_path: str, df_all: pd.DataFrame):
        self.model = self.load_model(model_path)
        self.df_all = df_all
        self.sensitive_features = {}
        self.metrics = {
            "Mean Squared Error": mean_squared_error,
            "Mean Absolute Error": mean_absolute_error
        }
        self.y_pred = None
        self.metric_frame = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_model(self, model_path: str):
        """Load the pre-trained model from a pickle file."""
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logging.info("Model loaded successfully from %s.", model_path)
        return model
        # try:
        #     with open(model_path, 'rb') as file:
        #         model = pickle.load(file)
        #     logging.info("Model loaded successfully from %s.", model_path)
        #     return model
        # except Exception as e:
        #     logging.error("Failed to load model from %s. Error: %s", model_path, e)
        #     raise

    def prepare_data(self, selected_df: pd.DataFrame):
        """Prepare the data by dropping the datetime column and splitting into train/test."""
        # Removing the 'datetime' column if it exists
        if 'datetime' in selected_df.columns:
            selected_df = selected_df.drop('datetime', axis=1)

        # Separate features and target variable
        X = selected_df.drop('value', axis=1)  # Features (dropping the target column 'value')
        y = selected_df['value']  # Target variable (the 'value' column)

        # Split the data: 80% for training, 20% for testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logging.info("Training set shape: %s, %s", self.X_train.shape, self.y_train.shape)
        logging.info("Test set shape: %s, %s", self.X_test.shape, self.y_test.shape)

    def generate_predictions(self):
        """Generate predictions using the loaded model."""
        self.y_pred = self.model.predict(self.X_test)
        logging.info("Predictions successfully generated for %d instances.", len(self.X_test))
        return self.y_pred

    def augment_test_data(self):
        """Add additional columns required for bias detection."""
        self.X_test['zone'] = self.df_all.loc[self.X_test.index, 'zone'].astype(str)
        self.X_test['subba-name'] = self.df_all.loc[self.X_test.index, 'subba-name'].astype(str)
        self.X_test['cloudcover'] = pd.to_numeric(self.df_all.loc[self.X_test.index, 'cloudcover'], errors='coerce')
        logging.info("Test data augmented with additional columns: zone, subba-name, cloudcover.")

    def set_sensitive_features(self):
        """Define sensitive features for bias detection."""
        self.sensitive_features = {
            "zone": self.X_test['zone'],
            "cloudcover_high_low": np.where(self.X_test['cloudcover'] > self.X_test['cloudcover'].median(), 'high', 'low')
        }
        logging.info("Sensitive features set up successfully: %s", list(self.sensitive_features.keys()))

    def evaluate_bias(self) -> pd.DataFrame:
        """Using Fairlearn's MetricFrame to calculate group-wise metrics."""
        self.metric_frame = MetricFrame(
            metrics=self.metrics,
            y_true=self.y_test,
            y_pred=self.y_pred,
            sensitive_features=self.sensitive_features
        )
        logging.info("Bias evaluation completed. Metrics calculated for sensitive features.")
        return self.metric_frame.by_group

    def get_group_metrics(self) -> pd.DataFrame:
        """Retrieve metrics by group."""
        logging.info("Retrieving group metrics.")
        return self.metric_frame.by_group


if __name__ == "__main__":
    # Set up argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Detect model bias in predictions.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model pickle file.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV data file containing all data.")
    
    args = parser.parse_args()

    # Load data
    try:
        df_all = pd.read_csv(args.data_path)
        logging.info("Data loaded successfully from %s.", args.data_path)
    except Exception as e:
        logging.error("Failed to load data from %s. Error: %s", args.data_path, e)
        raise

    # Initialize ModelBiasDetector
    detector = ModelBiasDetector(model_path=args.model_path, df_all=df_all)

    # Prepare data and run bias detection
    detector.prepare_data(df_all)
    detector.generate_predictions()
    detector.augment_test_data()
    detector.set_sensitive_features()
    
    # Evaluate and print group metrics
    group_metrics = detector.evaluate_bias()
    print(group_metrics)

# usage python data_bias_with_model.py --model_path path/to/model.pkl --data_path path/to/data.csv
