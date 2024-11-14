from datetime import datetime
import os
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from mlflow_utils import *
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV


# Load and preprocess data
data = pd.read_csv('/Users/amoghagadde/Desktop/Amogha/Northeastern/SEM_3/ML_Ops/Project/mlops-project/dataset/data/data_preprocess.csv')

# Split features and target
X = data.drop(columns=['value', 'datetime', 'zone'])
y = data[['value']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

# Label encoding for 'subba_name' - fitting only on training data
label_encoder = LabelEncoder()
X_train['subba_name_encoded'] = label_encoder.fit_transform(X_train['subba-name'])
X_test['subba_name_encoded'] = label_encoder.transform(X_test['subba-name'])

# Drop the original 'subba_name' column
X_train = X_train.drop(columns=['subba-name'])
X_test = X_test.drop(columns=['subba-name'])

# Define MLflow tags
tags = {
    "model_name": "XGBoost",
    "version": "v3.0",
    "purpose": "Model Selection",
    "iterations":2
}

# Generate timestamped run name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{tags['model_name']}_{tags['version']}_{timestamp}"

# Start MLflow run
set_tracking_uri("http://127.0.0.1:5001")
run = start_mlflow_run(run_name=run_name, tags=tags)

if run:
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

    # Hyperparameter tuning with RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb_reg,
        param_distributions=param_dist,
        scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
        refit='neg_mean_squared_error',
        cv=5,
        n_iter=2, 
        n_jobs=-1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train.values.ravel())
    best_model = random_search.best_estimator_

    # Display best parameters and cross-validation score
    print("Best parameters found:", random_search.best_params_)
    print("Best cross-validated MSE score:", -random_search.best_score_)
    
    # Display additional metrics
    results = random_search.cv_results_
    print("Mean Cross-Validated MAE:", -results['mean_test_neg_mean_absolute_error'][random_search.best_index_])
    print("Mean Cross-Validated R²:", results['mean_test_r2'][random_search.best_index_])

    # Test set evaluation
    y_test_pred = best_model.predict(X_test)
    print("Test Set Metrics:")
    print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_test_pred))
    print("R-squared (R²):", r2_score(y_test, y_test_pred))
    
    # Log model and metrics with MLflow
    predictions_xgb = best_model.predict(X_train)
    log_model(best_model, "XGBoost model")
    log_metric("MSE", mean_squared_error(y_test, y_test_pred))
    log_metric("MAE", mean_absolute_error(y_test, y_test_pred))
    log_metric("R2", r2_score(y_test, y_test_pred))

    # End MLflow run
    end_run()
else:
    print("MLflow run was not started. Check for errors.")