# usage - 
# python train.py path/to/dataset.csv --config config.json --model lr 
# python train.py path/to/dataset.csv --config config.json --model lstm 
# python train.py path/to/dataset.csv --config config.json --model xgboost 


import argparse
import os
import pickle
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.linear_model import LinearRegression  # Updated to LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV
import json
from data_loader import load_and_split_dataset
import logging
from datetime import datetime
from mlflow_utils import *

# Setting up logger
logger = logging.getLogger("ModelTrainer")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class ModelTrainer:
    def __init__(self, config_path=None, load_existing_model=False):
        if config_path == None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')

        self.config = self.load_config(config_path)

        # Set paths to local dataset folders
        self.train_data_path = os.path.join(os.path.dirname(__file__), '../data/train_data.csv')
        # self.validation_data_path = os.path.join(os.path.dirname(__file__), '../data/validate_data.csv')
        self.test_data_path = os.path.join(os.path.dirname(__file__), '../data/test_data.csv')

        #self.dataset_path = dataset_path
        self.features = self.config["features"]
        self.label = self.config["label"]
        self.test_size = self.config["test_size"]
        # self.validation_size = self.config["validation_size"]
        self.xgb_params = self.config["xgboost_param_dist"]
        # self.learning_rate = self.config["learning_rate"]
        self.load_existing_model = load_existing_model
        self.model_save_path = os.path.join(os.path.dirname(__file__), '../pickle/')

        # Create the folder if it doesn't exist
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    def load_dataset(self, dataset_path=None):
        if dataset_path != None:
            self.dataset_path = dataset_path

            try:
                # self.train_data, self.validation_data, self.test_data = load_and_split_dataset(
                #     dataset_path, self.test_size, self.validation_size, save_locally=False
                # )
                self.train_data, self.test_data = load_and_split_dataset(
                    dataset_path, self.test_size, save_locally=False
                )
            except FileNotFoundError as e:
                logger.error(f"Error loading dataset: {e}")
                raise
        
        else:
            try:
                # Load datasets from local folder paths
                self.train_data = pd.read_csv(self.train_data_path)
                # self.validation_data = pd.read_csv(self.validation_data_path)
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

    def preprocess_data(self, date=None):
        # Define features and label
        if date!=None:
            self.train_data = self.train_data[self.train_data['datetime'] >= date]

        X_train = self.train_data[self.features]
        y_train = self.train_data[self.label]
        # X_val = self.validation_data[self.features]
        # y_val = self.validation_data[self.label]
        X_test = self.test_data[self.features]
        y_test = self.test_data[self.label]

        
        # Standardizing the features
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_val = scaler.transform(X_val)
        # X_test = scaler.transform(X_test)
        
        # return X_train, X_val, X_test, y_train, y_val, y_test
        return X_train, X_test, y_train, y_test

    # def save_model(self, model, model_type, dataset_date):
    #     """Save the trained model using pickle with a timestamp."""
    #     # Get the current timestamp
    #     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
    #     # Construct the filename with timestamp
    #     model_filename = os.path.join(self.model_save_path, f"{model_type}_model_.pkl")

    #     # Store the model and the date in a dictionary
    #     model_data = {
    #         'model': model,
    #         'dataset_date': dataset_date
    #     }

    #     # Save the model to the pickle file
    #     with open(model_filename, 'wb') as f:
    #         pickle.dump(model_data, f)
        
    #     logger.info(f"Model saved to {model_filename}")

    # def load_model(self, model_type):
    #     """Load the most recent trained model based on timestamp."""
    #     # List all files in the pickle folder
    #     model_files = [f for f in os.listdir(self.model_save_path) if f.startswith(f"{model_type}_model")]
        
    #     if not model_files:
    #         logger.error(f"No models found for {model_type}.")
    #         return None
        
    #     # Sort the files by timestamp (filename format ensures correct sorting)
    #     model_files.sort(reverse=True)  # Most recent file first
        
    #     # Get the most recent model file
    #     most_recent_model_file = model_files[0]
        
    #     # Load the model from the most recent file
    #     model_filename = os.path.join(self.model_save_path, most_recent_model_file)
    #     with open(model_filename, 'rb') as f:
    #         model_data = pickle.load(f)
        
    #     model = model_data["model"]
    #     dataset_date = model_data["dataset_date"]
    #     logger.info(f"Model loaded from {model_filename}")
    #     return model, dataset_date

    # def train_lr(self, X_train, y_train, X_val, y_val, model=None):
    #     # Use the provided model or create a new one if none is given
    #     if model is None:
    #         model = LinearRegression()
    #         model.fit(X_train, y_train)
    #     else:
    #         model.fit(X_train, y_train)

    #     # Prediction and evaluation
    #     y_pred = model.predict(X_val)
    #     mse = mean_squared_error(y_val, y_pred)
    #     mae = mean_absolute_error(y_val, y_pred)
    #     r2 = r2_score(y_val, y_pred)

    #     return model, mse, mae, r2

    # def train_lstm(self, X_train, y_train, X_val, y_val):
    #     # Reshaping data for LSTM (samples, time steps, features)
    #     X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    #     X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

    #     model = Sequential()
    #     model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    #     model.add(Dropout(0.2))
    #     model.add(LSTM(50, activation='relu'))
    #     model.add(Dropout(0.2))
    #     model.add(Dense(1, activation='sigmoid'))

    #     optimizer = Adam(learning_rate=self.config["learning_rate"])
    #     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    #     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
    #     model.fit(X_train, y_train, epochs=self.config["epochs"], batch_size=self.config["batch_size"], validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
    #     loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        
    #     return model, accuracy, loss
    

    # def train_xgboost(self, X_train, y_train, X_val, y_val):
        # model = xgb.XGBRegressor(objective='reg:squarederror', 
        #                          random_state=42,
        #                          reg_alpha=0.1, 
        #                          reg_lambda=1.0,
        #                          learning_rate=self.config["learning_rate"], 
        #                          n_estimators=1000, 
        #                          min_child_weight=5)
        
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_val)
        # accuracy = accuracy_score(y_val, y_pred)
        
        # return model, accuracy
        # Define the XGBoost model
    def train_xgboost(self, X_train, y_train):
        
        # Define the model and parameter distribution for RandomizedSearchCV
        xgb_reg = XGBRegressor(objective='reg:squarederror', random_state=42)
        param_dist = self.config["xgboost_param_dist"]
        
        # Hyperparameter tuning with RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=xgb_reg,
            param_distributions=param_dist,
            scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
            refit='neg_mean_squared_error',
            cv=5,
            n_iter=20, 
            n_jobs=-1,
            random_state=42
        )
        
        random_search.fit(X_train, y_train.values.ravel())
        best_model = random_search.best_estimator_
        
        # Log best parameters and scores
        logger.info("Best parameters found: %s", random_search.best_params_)
        logger.info("Best cross-validated MSE score: %.4f", -random_search.best_score_)
        
        # Log additional metrics
        results = random_search.cv_results_
        logger.info("Mean Cross-Validated MAE: %.4f", -results['mean_test_neg_mean_absolute_error'][random_search.best_index_])
        logger.info("Mean Cross-Validated R²: %.4f", results['mean_test_r2'][random_search.best_index_])
 
        return best_model
        
    

    def evaluate_model(self, X_train, y_train, X_test, y_test):
        
        # Define MLflow tags
        tags = {
            "model_name": "XGBoost",
            "version": "v3.0",
            "purpose": "Model Selection",
            "iterations":20
        }

        # Generate timestamped run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{tags['model_name']}_{tags['version']}_{timestamp}"

        # Start MLflow run
        set_tracking_uri("http://127.0.0.1:5001")
        run = start_mlflow_run(run_name=run_name, tags=tags)

        if run:
            logger.info("Training XGBoost model...")
            model = self.train_xgboost(X_train, y_train)

            # Test set evaluation
            y_test_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_test_pred)
            mae = mean_absolute_error(y_test, y_test_pred)
            r2 = r2_score(y_test, y_test_pred)
            # Log test set metrics
            logger.info("Test Set Metrics:")
            logger.info("Mean Squared Error (MSE): %.4f", mse)
            logger.info("Mean Absolute Error (MAE): %.4f", mae)
            logger.info("R-squared (R²): %.4f", r2)
            
            # Log model and metrics with MLflow
            predictions_xgb = model.predict(X_train)
            log_model(model, "XGBoost model", X_train=X_train, predictions=predictions_xgb)
            log_metric("MSE", mse)
            log_metric("MAE", mae)
            log_metric("R2", r2)
            
            logger.info(f"Run {run.info.run_id} finished successfully!") 

            # End MLflow run
            end_run()
        else:
            logger.info("MLflow run was not started. Check for errors.")

        

    # def select_best_model(self, model_type, metric="R2"):
    #     """Selects the best model based on the specified metric from MLflow experiments."""
    #     client = MlflowClient()
        
    #     # Define experiment by name or use the active experiment ID
    #     experiment_id = client.get_experiment_by_name("Default").experiment_id
        
    #     # Query all runs for this experiment
    #     runs = client.search_runs(
    #         experiment_ids=[experiment_id],
    #         filter_string=f"params.model_type = '{model_type}'",
    #         order_by=[f"metrics.{metric} DESC"],  # Order by the specified metric, descending
    #         max_results=1  # Get only the top run
    #     )
        
    #     if runs:
    #         best_run = runs[0]
    #         best_run_id = best_run.info.run_id
    #         logger.info(f"Best {model_type} model selected with {metric} = {best_run.data.metrics[metric]} (Run ID: {best_run_id})")
            
    #         # Load the best model from MLflow
    #         if model_type == "lr":
    #             model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/linear_regression")
    #         elif model_type == "lstm":
    #             model = mlflow.tensorflow.load_model(f"runs:/{best_run_id}/lstm")
    #         elif model_type == "xgboost":
    #             model = mlflow.xgboost.load_model(f"runs:/{best_run_id}/xgboost")
            
    #         return model
    #     else:
    #         logger.warning(f"No runs found for model type '{model_type}' with metric '{metric}'")
    #         return None

def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Train models using different algorithms and track using MLflow.")
    parser.add_argument("path", type=str, nargs='?', default=None, help="Path to the dataset CSV file. Optional if using default paths.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration JSON file.")
    parser.add_argument("--model", type=str, choices=['lr', 'lstm', 'xgboost'], help="The type of model to train.")
    parser.add_argument("--load_existing_model", action='store_true', help="Flag to load an existing model instead of retraining.")
    
    args = parser.parse_args()

    trainer = ModelTrainer(args.config, args.load_existing_model)
    trainer.load_dataset(args.path)
    trainer.evaluate_model(args.model)

if __name__ == "__main__":
    main()

