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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import json
from data_loader import load_and_split_dataset
import logging
from datetime import datetime
import joblib

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
        self.model_save_path = os.path.join(os.path.dirname(__file__), '/../pickle/')

        # Create the folder if it doesn't exist
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

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

    def save_model(self, model, model_type, dataset_date):
        """Save the trained model using pickle with a timestamp. Saves to local folder"""
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
            model_data = pickle.load(f)
        
        model = model_data["model"]
        dataset_date = model_data["dataset_date"]
        logger.info(f"Model loaded from {model_filename}")
        return model, dataset_date
    
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

    def train_xgboost(self, X_train, y_train, X_val, y_val, experiment=1):
        if experiment == 0:
            model = xgb.XGBRegressor(objective='reg:squarederror', 
                                     random_state=42,
                                     reg_alpha=0.1, 
                                     reg_lambda=1.0,
                                     learning_rate=self.config["learning_rate"], 
                                     n_estimators=1000, 
                                     min_child_weight=5)
            
            model.fit(X_train, y_train)

        else:
            model_exp = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            param_dist = {
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 500, 1000],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.5, 1.0, 2.0]
            }

            random_search = RandomizedSearchCV(
                estimator=model_exp,
                param_distributions=param_dist,
                scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
                refit='neg_mean_squared_error',
                cv=5,
                n_iter=2, 
                n_jobs=-1,
                random_state=42
            )

            random_search.fit(X_train, y_train.values.ravel())
            model = random_search.best_estimator_

            logger.info("Best parameters found:", random_search.best_params_)
            logger.info("Best cross-validated MSE score:", -random_search.best_score_)

            # Display additional metrics
            results = random_search.cv_results_
            logger.info("Mean Cross-Validated MAE:", -results['mean_test_neg_mean_absolute_error'][random_search.best_index_])
            logger.info("Mean Cross-Validated R²:", results['mean_test_r2'][random_search.best_index_])

        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        return model, accuracy

    def train(self, model_type):
        # Start an MLflow run
        mlflow.set_tracking_uri(self.config["mlflow_tracking_uri"])

        with mlflow.start_run():
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
                mlflow.log_metric("MSE", mse)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("R2", r2)
                mlflow.sklearn.log_model(model, "linear_regression")

            elif model_type == 'lstm':
                logger.info("Training LSTM model...")
                model, lstm_accuracy, lstm_loss = self.train_lstm(X_train, y_train, X_val, y_val)
                y_test_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_test_pred)
                mae = mean_absolute_error(y_test, y_test_pred)
                r2 = r2_score(y_test, y_test_pred)
                
                mlflow.log_metric("accuracy", lstm_accuracy)
                mlflow.log_metric("loss", lstm_loss)
                mlflow.log_metric("MSE", mse)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("R2", r2)
                mlflow.tensorflow.log_model(model, "lstm")

            elif model_type == 'xgboost':
                logger.info("Training XGBoost model...")
                model, xgboost_accuracy = self.train_xgboost(X_train, y_train, X_val, y_val)
                mlflow.log_metric("accuracy", xgboost_accuracy)
                mlflow.xgboost.log_model(model, "xgboost")

            logger.info(f"{model_type} model trained and logged with MLflow.")
            
            # Save the model to disk after training
            # latest_date = self.train_data['datetime'].max()
            # self.save_model(model, model_type, latest_date)

    def select_best_model(self, model_type, metric="R2"):
        """Selects the best model based on the specified metric from MLflow experiments."""
        client = MlflowClient()
        
        # Define experiment by name or use the active experiment ID
        experiment_id = client.get_experiment_by_name("Default").experiment_id
        
        # Query all runs for this experiment
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"params.model_type = '{model_type}'",
            order_by=[f"metrics.{metric} DESC"],  # Order by the specified metric, descending
            max_results=1  # Get only the top run
        )
        
        if runs:
            best_run = runs[0]
            best_run_id = best_run.info.run_id
            logger.info(f"Best {model_type} model selected with {metric} = {best_run.data.metrics[metric]} (Run ID: {best_run_id})")
            
            # Load the best model from MLflow
            if model_type == "lr":
                model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/linear_regression")
            elif model_type == "lstm":
                model = mlflow.tensorflow.load_model(f"runs:/{best_run_id}/lstm")
            elif model_type == "xgboost":
                model = mlflow.xgboost.load_model(f"runs:/{best_run_id}/xgboost")
            
            return model
        else:
            logger.warning(f"No runs found for model type '{model_type}' with metric '{metric}'")
            return None

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

