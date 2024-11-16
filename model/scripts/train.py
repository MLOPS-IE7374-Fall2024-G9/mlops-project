# usage - 
# python train.py path/to/dataset.csv --config config.json --model lr 
# python train.py path/to/dataset.csv --config config.json --model lstm 
# python train.py path/to/dataset.csv --config config.json --model xgboost 


import argparse
import os
import pickle
import pandas as pd
import json
import logging
from datetime import datetime
import joblib
import subprocess

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from sklearn.linear_model import LinearRegression  # Updated to LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from model.scripts.utils import *
from model.scripts.data_loader import load_and_split_dataset

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
        self.validation_data_path = os.path.join(os.path.dirname(__file__), '../data/validate_data.csv')
        self.test_data_path = os.path.join(os.path.dirname(__file__), '../data/test_data.csv')

        #self.dataset_path = dataset_path
        self.features = self.config["features"]
        self.label = self.config["label"]
        self.test_size = self.config["test_size"]
        self.validation_size = self.config["validation_size"]
        self.learning_rate = self.config["learning_rate"]
        self.load_existing_model = load_existing_model
        self.model_save_path = os.path.dirname(__file__) + '/../pickle/'
        self.model = None

        # Create the folder if it doesn't exist
        # if not os.path.exists(self.model_save_path):
        #     os.makedirs(self.model_save_path, exist_ok=True)

        self.configure_mlfow_credentials("mlops-7374-3e7424e80d76.json")

    def configure_mlfow_credentials(self, json_credential_path):
        """
        Function to run the mflow credentials
        """
        try:
            # Run the dvc remote modify command
            logger.info(f"Configuring mlflow credentials from {json_credential_path}.")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=json_credential_path
            logger.info("MLflow remote configuration successful.")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to configure mflow remote: {e}")
    
    def setup_mlflow(self, model_name):
        # Define MLflow tags
        tags = {
            "model_name": model_name,
            "version": "v3.0",
            "purpose": "Model Selection",
            "iterations":2
        }

        # Generate timestamped run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{tags['model_name']}_{tags['version']}_{timestamp}"

        run = start_mlflow_run(run_name, tags)

        return run

    def load_dataset(self, dataset_path=None):
        if dataset_path != None:
            self.dataset_path = dataset_path

            try:
                self.train_data, self.validation_data, self.test_data, _, _ = load_and_split_dataset(
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

    def preprocess_data(self, date=None):
        # Define features and label
        if date!=None:
            self.train_data = self.train_data[self.train_data['datetime'] >= date]

        X_train = self.train_data[self.features]
        y_train = self.train_data[self.label]
        X_val = self.validation_data[self.features]
        y_val = self.validation_data[self.label]
        X_test = self.test_data[self.features]
        y_test = self.test_data[self.label]

        label_encoder = LabelEncoder()
        label_encoder.fit(X_train['subba-name'])

        X_train['subba-name'] = label_encoder.transform(X_train['subba-name'])
        X_val['subba-name'] = label_encoder.transform(X_val['subba-name'])
        X_test['subba-name'] = label_encoder.transform(X_test['subba-name'])

        joblib.dump(label_encoder, os.path.join(os.path.dirname(__file__)) + '/../pickle/label_encoder_subba-name.pkl')

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_model(self, model, model_type, dataset_date=None):
        """Save the trained model using pickle with a timestamp. Saves to local folder"""
        if dataset_date != None:
            # Get the current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Construct the filename with timestamp
            model_filename = os.path.join(self.model_save_path, f"{model_type}_model_.pkl")

            # Store the model and the date in a dictionary
            model_data = {
                'model': model,
                'dataset_date': dataset_date
            }

            # Save the model to the pickle file
            with open(model_filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {model_filename}")
        else:
            model_filename = os.path.join(self.model_save_path, f"{model_type}_model_.pkl")
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)

            logger.info(f"Model saved to {model_filename}")

    def load_model(self, model_type):
        """Load the most recent trained model based on timestamp. This loads from local folder"""
        # List all files in the pickle folder
        model_files = [f for f in os.listdir(self.model_save_path) if f.startswith(f"{model_type}_model")]
        
        if not model_files:
            logger.error(f"No models found for {model_type}.")
            return None
        
        # Sort the files by timestamp (filename format ensures correct sorting)
        model_files.sort(reverse=True)  # Most recent file first
        
        # Get the most recent model file
        most_recent_model_file = model_files[0]
        
        # Load the model from the most recent file
        model_filename = os.path.join(self.model_save_path, most_recent_model_file)
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
        
        #model = model_data["model"]
        #dataset_date = model_data["dataset_date"]
        logger.info(f"Model loaded from {model_filename}")
        self.model = model
        return model
    
    def load_model_artifact(self, model_type):
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
        self.model = model
        return model
    
    def load_model_mlflow(self, model_type):
        """Loads the model based on the type from the mlflow upstream server"""
        pass

    def train_lr(self, X_train, y_train, X_val, y_val, model=None):
        # Use the provided model or create a new one if none is given
        if model is None:
            model = LinearRegression()
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        # Prediction and evaluation
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        return model, mse, mae, r2

    def train_lstm(self, X_train, y_train, X_val, y_val):
        # Reshaping data for LSTM (samples, time steps, features)
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(learning_rate=self.config["learning_rate"])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(X_train, y_train, epochs=self.config["epochs"], batch_size=self.config["batch_size"], validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        return model, accuracy, loss

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        model_exp = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

        param_dist = self.config["xgboost_param_dist"]

        random_search = RandomizedSearchCV(
            estimator=model_exp,
            param_distributions=param_dist,
            scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
            refit='neg_mean_squared_error',
            cv=5,
            n_iter=self.config["xgboost_iter"],
            n_jobs=-1,
            random_state=42
        )

        # XGBoost supports evaluation sets for monitoring progress
        eval_set = [(X_train, y_train), (X_val, y_val)]

        # Fit the RandomizedSearchCV model
        random_search.fit(
            X_train,
            y_train.values.ravel(),
            eval_set=eval_set,
            eval_metric="r2",
            verbose=True  # Enables logging of progress
        )

        # Get the best model
        model = random_search.best_estimator_

        # Log the final model with MLflow
        predictions_xgb = model.predict(X_train)
        log_model(model, "XGBoost model", X_train=X_train, predictions=predictions_xgb)

        return model

    def train(self, model_type, save_local=False):
        # Start an MLflow run
        mlflow.set_tracking_uri(self.config["mlflow_tracking_uri"])
        run = self.setup_mlflow(model_type)
        mlflow.autolog()
        #with mlflow.start_run():
        if run:
            # Check if we need to load an existing model
            if self.load_existing_model:
                model, date = self.load_model(model_type)
                if model is None:
                    logger.error(f"Failed to load existing model for {model_type}. Starting fresh training.")
                    model = None
                    raise
                else:
                    # Preprocess the data
                    X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(date)
            else:
                model = None
                # Preprocess the data
                X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data()

            # Train the selected model
            if model_type == 'lr':
                logger.info("Training Linear Regression model...")
                model, mse, mae, r2 = self.train_lr(X_train, y_train, X_val, y_val, model)
                # mlflow.log_metric("MSE", mse)
                # mlflow.log_metric("MAE", mae)
                # mlflow.log_metric("R2", r2)
                mlflow.sklearn.log_model(model, "linear_regression")

            elif model_type == 'lstm':
                logger.info("Training LSTM model...")
                model, lstm_accuracy, lstm_loss = self.train_lstm(X_train, y_train, X_val, y_val)
                y_test_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_test_pred)
                mae = mean_absolute_error(y_test, y_test_pred)
                r2 = r2_score(y_test, y_test_pred)
                
                # mlflow.log_metric("accuracy", lstm_accuracy)
                # mlflow.log_metric("loss", lstm_loss)
                # mlflow.log_metric("MSE", mse)
                # mlflow.log_metric("MAE", mae)
                # mlflow.log_metric("R2", r2)
                mlflow.tensorflow.log_model(model, "lstm")

            elif model_type == 'xgboost':
                logger.info("Training XGBoost model...")
                model = self.train_xgboost(X_train, y_train, X_val, y_val)

            logger.info(f"{model_type} model trained and logged with MLflow.")
            
            if save_local:
                # Save the model to disk after training
                # latest_date = self.train_data['datetime'].max()
                self.save_model(model, model_type)

            end_run()
        else:
            logger.error("Error in starting MLflow run")
        
        
    
    def evaluate(self, model=None):
        _, _, X_test, _, _, y_test = self.preprocess_data()
        
        if model == None and self.model!=None:
            model = self.model
        elif model == None and self.model==None:
            raise("Model is none")

        y_test_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        
        # Log test set metrics
        logger.info("Test Set Metrics:")
        logger.info("Mean Squared Error (MSE): %.4f", mse)
        logger.info("Mean Absolute Error (MAE): %.4f", mae)
        logger.info("R-squared (R²): %.4f", r2)

        log_metric("MSE", mse)
        log_metric("MAE", mae)
        log_metric("R2", r2)

        return mse, mae, r2

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
    trainer.train(args.model)

if __name__ == "__main__":
    main()
