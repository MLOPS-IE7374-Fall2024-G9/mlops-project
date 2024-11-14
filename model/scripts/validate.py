import argparse
import os
import pickle
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from dataset_loader import load_and_split_dataset
import logging
import json

# Setting up logger
logger = logging.getLogger("ModelValidator")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class ModelValidator:
    def __init__(self, config_path=None, model_type="xgboost"):
        if config_path == None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')

        # Set paths to local dataset folders
        self.train_data_path = os.path.join(os.path.dirname(__file__), '../data/train_data.csv')
        self.validation_data_path = os.path.join(os.path.dirname(__file__), '../data/validate_data.csv')
        self.test_data_path = os.path.join(os.path.dirname(__file__), '../data/test_data.csv')

        self.config = self.load_config(config_path)
        self.test_size = self.config["test_size"]
        self.validation_size = self.config["validation_size"]
        self.model_type = model_type
        self.model_save_path = os.path.join(os.path.dirname(__file__), '../pickle')
        
        # Load the dataset
        self.load_dataset()

        # Ensure that the model file exists
        self.model = self.load_model()

    def load_dataset(self, dataset_path=None):
        if dataset_path != None:
            self.dataset_path = dataset_path

            try:
                self.train_data, self.validation_data, self.test_data = load_and_split_dataset(
                    dataset_path, self.test_size, self.validation_size, save_locally=False
                )
            except FileNotFoundError as e:
                logger.error(f"Error loading dataset: {e}")
                raise
        
        else:
            try:
                # Load datasets from local folder paths
                self.train_data = pd.read_csv(self.train_data_path)
                self.validation_data = pd.read_csv(self.validation_data_path)
                self.test_data = pd.read_csv(self.test_data_path)
            except FileNotFoundError as e:
                logger.error(f"Error loading dataset: {e}")
                raise

    def load_config(self, config_path):
        """Load configuration from a JSON file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
        
        with open(config_path, 'r') as file:
            config = json.load(file)
        
        return config

    def load_model(self):
        """Load the most recent model based on model type and timestamp."""
        model_files = [f for f in os.listdir(self.model_save_path) if f.startswith(f"{self.model_type}_model")]
        
        if not model_files:
            logger.error(f"No models found for {self.model_type}.")
            raise FileNotFoundError(f"No saved model found for {self.model_type}.")
        
        model_files.sort(reverse=True)  # Sort by timestamp (most recent first)
        model_filename = os.path.join(self.model_save_path, model_files[0])

        # Load the model
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from {model_filename}")
        return model

    def preprocess_data(self):
        """Preprocess the test data for evaluation."""
        X_test = self.test_data[self.config["features"]]
        y_test = self.test_data[self.config["label"]]

        # Standardizing the features using the scaler from training
        scaler = StandardScaler()
        X_test = scaler.fit_transform(X_test)

        return X_test, y_test

    def evaluate(self):
        """Evaluate the loaded model on the test dataset."""
        X_test, y_test = self.preprocess_data()

        # Start MLflow run for logging metrics
        with mlflow.start_run():
            # Evaluate the model based on its type
            if self.model_type == 'lr':
                logger.info("Evaluating Linear Regression model...")
                y_pred = self.model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                mlflow.log_metric("MSE", mse)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("R2", r2)
                logger.info(f"Evaluation results - MSE: {mse}, MAE: {mae}, R2: {r2}")

            elif self.model_type == 'lstm':
                logger.info("Evaluating LSTM model...")
                y_pred = self.model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                mlflow.log_metric("MSE", mse)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("R2", r2)
                logger.info(f"Evaluation results - MSE: {mse}, MAE: {mae}, R2: {r2}")

            elif self.model_type == 'xgboost':
                logger.info("Evaluating XGBoost model...")
                y_pred = self.model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                mlflow.log_metric("accuracy", accuracy)
                logger.info(f"Evaluation results - Accuracy: {accuracy}")

            else:
                logger.error(f"Unknown model type: {self.model_type}")
                raise ValueError(f"Unknown model type: {self.model_type}")

def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Validate models using different algorithms and evaluate performance.")
    parser.add_argument("path", type=str, nargs='?', default=None, help="Path to the dataset CSV file. Optional if using default paths.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration JSON file.")
    parser.add_argument("--model", type=str, choices=['lr', 'lstm', 'xgboost'], help="The type of model to evaluate.")
    
    args = parser.parse_args()

    validator = ModelValidator(args.config, args.model)
    validator.load_dataset()
    validator.evaluate()

if __name__ == "__main__":
    main()
