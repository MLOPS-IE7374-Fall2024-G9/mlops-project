# Import the mlflow functions from mlflow.py
# import mlflow_utils as ml_utils
from datetime import datetime
import os
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from mlflow_utils import *
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV


def set_tracking_uri(uri):
    mlflow.set_tracking_uri(uri)
  
 
def start_mlflow_run(run_name=None, tags=None):
    client = MlflowClient()
    experiment_name = "Electricity Demand Prediction_1"

 
    try:
        # Check if the experiment exists
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' does not exist. Creating a new one.")
            experiment_id = client.create_experiment(experiment_name)
            print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            print(f"Found existing experiment '{experiment_name}' with ID: {experiment_id}")
 
        # Set the experiment
        mlflow.set_experiment(experiment_name)
        print(f"Setting active experiment to '{experiment_name}'.")
 
        # Start the run
        run = mlflow.start_run(run_name=run_name)
        print(f"Started run '{run_name}' with ID: {run.info.run_id}")
 
        # Add tags if provided
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
                print(f"Set tag '{key}' to '{value}'.")
 
        return run
 
    except Exception as e:
        print(f"Error in starting MLflow run: {e}")
 
# Function to log a metric
def log_metric(metric_name, value, step=None):
    try:
        mlflow.log_metric(metric_name, value, step)
        print(f"Logged metric '{metric_name}': {value}")
    except Exception as e:
        print(f"Error logging metric '{metric_name}': {e}")
 
# Function to log a parameter
def log_param(param_name, value):
    try:
        mlflow.log_param(param_name, value)
        print(f"Logged parameter '{param_name}': {value}")
    except Exception as e:
        print(f"Error logging parameter '{param_name}': {e}")
 
# Function to end the MLflow run
def end_run():
    try:
        mlflow.end_run()
        print("Ended the MLflow run.")
    except Exception as e:
        print(f"Error ending the MLflow run: {e}")


data= pd.read_csv('/Users/amoghagadde/Desktop/Amogha/Northeastern/SEM_3/ML_Ops/Project/bias_mitigated_data.csv')
data = data.dropna()
X = data.drop(columns=['value', 'datetime'])
y = data[['value']]

# X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2)
 
# Define tags
tags = {
    "model_name": "XGBoost",
    "version": "v1.0",
    # "dataset_version": "v2",
    "purpose": "Model Selection"
}
 
# Get the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 
# Create a descriptive run name with the timestamp
run_name = f"{tags['model_name']}_{tags['version']}_{timestamp}"
 
# Start the MLflow run with the descriptive run name and tags
# set_tracking_uri("http://127.0.0.1:5000")
# run = start_mlflow_run(run_name=run_name, tags=tags)
 

# if run:

#     from xgboost import XGBRegressor
#     from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#     from sklearn.model_selection import cross_val_score
    
#     # Initialize the XGBoost model with default hyperparameters
#     xgb_reg = XGBRegressor(objective='reg:squarederror', random_state=42,reg_alpha=0.1, reg_lambda=1.0,learning_rate=0.05, n_estimators=1000, min_child_weight=5)


#     # Perform 5-fold cross-validation
#     scores = cross_val_score(xgb_reg, X, y, cv=5, scoring='r2')
#     print("Cross-Validated R² scores:", scores)
#     print("Mean Cross-Validated R²:", np.mean(scores))

#     # Mean Absolute Error (MAE)
#     mae_scores = cross_val_score(xgb_reg, X, y, cv=5, scoring='neg_mean_absolute_error')
#     mae_scores = -mae_scores  # Convert to positive MAE values
#     print("Cross-Validated MAE scores:", mae_scores)
#     print("Mean Cross-Validated MAE:", np.mean(mae_scores))

#     # Mean Squared Error (MSE)
#     mse_scores = cross_val_score(xgb_reg, X, y, cv=5, scoring='neg_mean_squared_error')
#     mse_scores = -mse_scores  # Convert to positive MSE values
#     print("Cross-Validated MSE scores:", mse_scores)
#     print("Mean Cross-Validated MSE:", np.mean(mse_scores))


#     X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2)

#     # Fit the model on the training data
#     xgb_reg.fit(X_train, y_train)

#     # Make predictions on the test data
#     y_test_pred = xgb_reg.predict(X_test)
    
#     # Infer signature
#     predictions_xgb = xgb_reg.predict(X_train)
#     # signature_xgb = infer_signature(X_train, y_test_pred)

#     # Logging the XGBoost model with the signature
#     # mlflow.xgboost.log_model(xgb_reg, "XGBoost model", signature=signature_xgb)
#     log_model(xgb_reg, "XGBoost model", X_train=X_train, predictions = predictions_xgb)

#     # Evaluate the model on the test set
#     print("XGBoost Regression Test Set Metrics:")
#     print("Mean Squared Error (MSE):", np.mean(mse_scores))
#     print("Mean Absolute Error (MAE):", np.mean(mae_scores))
#     print("R-squared (R²):", np.mean(scores))


#     # Set parameters and log metrics
#     log_metric("MSE", np.mean(mse_scores))
#     log_metric("MAE", np.mean(mae_scores))
#     log_metric("R2", np.mean(scores))

#     # End the run
#     end_run()

# else:
#     print("MLflow run was not started. Check for errors.")

# Initial train-test split before any model training or hyperparameter tuning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model and parameter distribution for RandomizedSearchCV
xgb_reg = XGBRegressor(objective='reg:squarederror', random_state=42)
param_dist = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1.0, 2.0]
}

# Perform hyperparameter tuning on the training set only
random_search = RandomizedSearchCV(
    estimator=xgb_reg,
    param_distributions=param_dist,
    scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],  # Track multiple metrics
    refit='neg_mean_squared_error',  # Refit using MSE as primary metric
    cv=5,
    n_iter=30,
    n_jobs=-1,
    random_state=42
)

# Fit RandomizedSearchCV on the training set
random_search.fit(X_train, y_train, verbose=True)

# Extract the best model
best_model = random_search.best_estimator_

# Display the best parameters and the cross-validated score
print("Best parameters found:", random_search.best_params_)
print("Best cross-validated MSE score:", -random_search.best_score_)

# Display cross-validation scores for each metric
results = random_search.cv_results_
print("Mean Cross-Validated MAE:", -results['mean_test_neg_mean_absolute_error'][random_search.best_index_])
print("Mean Cross-Validated R²:", results['mean_test_r2'][random_search.best_index_])

# Evaluate the best model on the separate test set
y_test_pred = best_model.predict(X_test)
print("Test Set Metrics:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_test_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_test_pred))
print("R-squared (R²):", r2_score(y_test, y_test_pred))