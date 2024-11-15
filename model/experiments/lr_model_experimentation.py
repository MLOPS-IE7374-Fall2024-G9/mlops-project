from datetime import datetime
import os
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from mlflow_utils import *



data= pd.read_csv('/Users/amoghagadde/Desktop/Amogha/Northeastern/SEM_3/ML_Ops/Project/mlops-project/dataset/data/data_preprocess.csv')
# data = data.dropna()

X = data.drop(columns=['value', 'datetime', 'zone'])
y = data[['value']]

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2)

# Label encoding for 'subba_name' - fitting only on training data
label_encoder = LabelEncoder()
X_train['subba_name_encoded'] = label_encoder.fit_transform(X_train['subba-name'])
X_test['subba_name_encoded'] = label_encoder.transform(X_test['subba-name'])

# Drop the original 'subba_name' column
X_train = X_train.drop(columns=['subba-name'])
X_test = X_test.drop(columns=['subba-name'])
 
# Define tags
tags = {
    "model_name": "Linear Regression",
    "version": "v3.0",
    # "dataset_version": "v2",
    "purpose": "Model Selection"
}
 
# Get the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 
# Create a descriptive run name with the timestamp
run_name = f"{tags['model_name']}_{tags['version']}_{timestamp}"
 
# Start the MLflow run with the descriptive run name and tags
set_tracking_uri("http://127.0.0.1:5001")
# set_tracking_uri("http://34.56.170.84:5000")
run = start_mlflow_run(run_name=run_name, tags=tags)

if run: 
        # # Linear Regression Tracking
        # with mlflow.start_run(run_name="Linear Regression") as run:
        
        # Debugging: Check active run
        print(f"Active run_id: {run.info.run_id}")
        
        # Initialize the Linear Regression model
        lin_reg = LinearRegression()
    
        # Fit the model on the training data
        lin_reg.fit(X_train, y_train)
    
        # Make predictions on the test data
        y_test_pred = lin_reg.predict(X_test)
        
        # Evaluate model performance
        mse = mean_squared_error(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
    
        # Evaluate the model on the test set
        print("Linear Regression Test Set Metrics:")
        print("Mean Squared Error (MSE):", mse)
        print("Mean Absolute Error (MAE):", mae)
        print("R-squared (R²):", r2)
        
        # Logging parameters, metrics, and model
        log_metric("MSE", mse)
        log_metric("MAE", mae)
        log_metric("R2", r2)
        #mlflow.sklearn.log_model(lin_reg, "Linear Regression model")
        
        # Debugging: Tracking URI to ensure it's set correctly
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print("Tracking URL type: ", tracking_url_type_store)

        # Infer signature
        predictions_lin = lin_reg.predict(X_train)
        # signature_lin = infer_signature(X_train, predictions_lin)

        # Log the model using the new function from mlflow_utils.py
        log_model(lin_reg, "Linear Regression model", X_train=X_train, predictions = predictions_lin)

        # Debugging: Confirm run status
        print(f"Run {run.info.run_id} finished successfully!")

        # End the run
        end_run()

else:
    print("MLflow run was not started. Check for errors.")
        
    