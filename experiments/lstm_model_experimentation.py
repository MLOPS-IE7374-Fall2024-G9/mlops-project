# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
# import pandas as pd

# data= pd.read_csv('/Users/akm/Desktop/mlops-project/preprocessed_data.csv')
# X = data.drop(columns=['value'])
# y = data[['value']]

# X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2)
 
# X_train = X_train.to_numpy().reshape((X_train.shape[0], 1, X_train.shape[1]))
# X_test = X_test.to_numpy().reshape((X_test.shape[0], 1, X_test.shape[1]))
 
# y_train = y_train.to_numpy()
# y_test = y_test.to_numpy()
 
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import mlflow
# import mlflow.sklearn
# import mlflow.keras
# from mlflow.models import infer_signature
# from urllib.parse import urlparse


# mlflow.set_tracking_uri("http://localhost:5000")
 
# # Experiment name
# # mlflow.create_experiment("Electricity Demand Prediction_1")
# # mlflow.set_experiment("Electricity Demand Prediction_1")
# # mlflow.create_experiment("Electricity Demand Prediction_1", artifact_location="/Users/amoghagadde/Desktop/Amogha/Northeastern/SEM_3/ML_Ops/Project/mlops-project/experiments")

# experiment = mlflow.get_experiment_by_name("Electricity Demand Prediction_1")
# if experiment is None:
#     mlflow.create_experiment("Electricity Demand Prediction_1")
# mlflow.set_experiment("Electricity Demand Prediction_1")

# mlflow.set_experiment("Electricity Demand Prediction_1")

# print("Current artifact location:", experiment.artifact_location)
 
# with mlflow.start_run(run_name = "LSTM") as run:
    
#     # Debugging: Check active run
#     print(f"Active run_id: {run.info.run_id}")
 
#     # Define n_timesteps and n_features
#     n_timesteps = X_train.shape[1]  
#     n_features = X_train.shape[2]   
 
#     # Model building
#     lstm_model = Sequential()
 
#     # Add the LSTM layers and dropout layers
#     lstm_model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features)))
#     lstm_model.add(Dropout(0.2))
#     lstm_model.add(LSTM(50, activation='relu'))
#     lstm_model.add(Dropout(0.2))
 
#     # Output layer
#     lstm_model.add(Dense(1))
 
#     # Compiling the model with Adam optimizer and MSE loss
#     optimizer = Adam(learning_rate=0.001)
#     lstm_model.compile(optimizer=optimizer, loss='mse')
 
#     # Early stopping to avoid overfitting
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
 
#     # Fit the model
#     lstm_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
 
#     # Making predictions on the test set
#     y_test_pred = lstm_model.predict(X_test)
 
#     # Evaluate model performance
#     mse = mean_squared_error(y_test, y_test_pred)
#     mae = mean_absolute_error(y_test, y_test_pred)
#     r2 = r2_score(y_test, y_test_pred)
 
#     print(f'Mean Squared Error (MSE): {mse}')
#     print(f'Mean Absolute Error (MAE): {mae}')
#     print(f'R-squared (R²): {r2}')
    
#     #Logging parameters, metrics, and model
#     mlflow.log_metric("MSE", mse)
#     mlflow.log_metric("MAE", mae)
#     mlflow.log_metric("R2", r2)
    
#     # Debugging: Tracking URI to ensure it's set correctly
#     tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
#     print("Tracking URL type: ", tracking_url_type_store)
    
#     # Infer signature
#     predictions_lstm = lstm_model.predict(X_train)
#     signature_lstm = infer_signature(X_train, predictions_lstm)
    
#     # Logging the model with signature
#     mlflow.keras.log_model(lstm_model, "LSTM model", signature = signature_lstm)
    
#     # Debugging: Confirm run status
#     print(f"Run {run.info.run_id} finished successfully!") 
    
    
    ########################################################################################################################
    
# Import the mlflow functions from mlflow.py
# import mlflow_utils as ml_utils
import pandas as pd
import numpy as np
from datetime import datetime
import os
# import mlflow
# import mlflow.sklearn
# import mlflow.keras
# from mlflow import MlflowClient
# from mlflow.models import infer_signature
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from mlflow_utils import *



# def set_tracking_uri(uri):
#     mlflow.set_tracking_uri(uri)
  
 
# def start_mlflow_run(run_name=None, tags=None):
#     client = MlflowClient()
#     experiment_name = "Electricity Demand Prediction_1"

 
#     try:
#         # Check if the experiment exists
#         experiment = client.get_experiment_by_name(experiment_name)
#         if experiment is None:
#             print(f"Experiment '{experiment_name}' does not exist. Creating a new one.")
#             experiment_id = client.create_experiment(experiment_name)
#             print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
#         else:
#             experiment_id = experiment.experiment_id
#             print(f"Found existing experiment '{experiment_name}' with ID: {experiment_id}")
 
#         # Set the experiment
#         mlflow.set_experiment(experiment_name)
#         print(f"Setting active experiment to '{experiment_name}'.")
 
#         # Start the run
#         run = mlflow.start_run(run_name=run_name)
#         print(f"Started run '{run_name}' with ID: {run.info.run_id}")
 
#         # Add tags if provided
#         if tags:
#             for key, value in tags.items():
#                 mlflow.set_tag(key, value)
#                 print(f"Set tag '{key}' to '{value}'.")
 
#         return run
 
#     except Exception as e:
#         print(f"Error in starting MLflow run: {e}")
 
# # Function to log a metric
# def log_metric(metric_name, value, step=None):
#     try:
#         mlflow.log_metric(metric_name, value, step)
#         print(f"Logged metric '{metric_name}': {value}")
#     except Exception as e:
#         print(f"Error logging metric '{metric_name}': {e}")
 
# # Function to log a parameter
# def log_param(param_name, value):
#     try:
#         mlflow.log_param(param_name, value)
#         print(f"Logged parameter '{param_name}': {value}")
#     except Exception as e:
#         print(f"Error logging parameter '{param_name}': {e}")
 
# # Function to end the MLflow run
# def end_run():
#     try:
#         mlflow.end_run()
#         print("Ended the MLflow run.")
#     except Exception as e:
#         print(f"Error ending the MLflow run: {e}")


data= pd.read_csv('/Users/amoghagadde/Desktop/Amogha/Northeastern/SEM_3/ML_Ops/Project/bias_mitigated_data.csv')
data = data.dropna()
X = data.drop(columns=['value', 'datetime'])
y = data[['value']]

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2)
 
X_train = X_train.to_numpy().reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.to_numpy().reshape((X_test.shape[0], 1, X_test.shape[1]))
 
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# # Convert X_train to DataFrame
# X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[2])
# X_train_df = pd.DataFrame(X_train_reshaped)

# # Check for NaN values in each column
# nan_counts = X_train_df.isna().sum()
# print(nan_counts)
    
# # Check for NaN values in y_test and y_test_pred
# print("NaN in X_train:", np.isnan(X_train).any())
# print("NaN in y_train:", np.isnan(y_train).any())
# print("NaN in X_test:", np.isnan(X_test).any())
# print("NaN in y_test:", np.isnan(y_test).any())
# # print("NaN in y_test_pred:", np.isnan(y_test_pred).any())

# # Optionally, check the count of NaN values
# print("NaN values in X_train:", np.isnan(X_train).sum())
# print("NaN values in y_train:", np.isnan(y_train).sum())
# print("NaN values in X_test:", np.isnan(X_test).sum())
# print("NaN values in y_test:", np.isnan(y_test).sum())
# # print("NaN values in y_test_pred:", np.isnan(y_test_pred).sum())


# import numpy as np
# print("Non-numeric values in X_train:", np.where(np.array([isinstance(x, (int, float)) for x in X_train.ravel()]) == False))
# print(data.columns)

# print(X_train.dtypes)
# print(y_train.dtypes)
# print(X_train.shape)
# print(y_train.shape)

# X_train = X_train.astype('float64')
# y_train = y_train.astype('float64')

# X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2)
 
# Define tags
tags = {
    "model_name": "LSTM",
    "version": "v2.0",
    # "dataset_version": "v2",
    "purpose": "Model Selection"
}
 
# Get the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 
# Create a descriptive run name with the timestamp
run_name = f"{tags['model_name']}_{tags['version']}_{timestamp}"
 
# Start the MLflow run with the descriptive run name and tags
set_tracking_uri("http://127.0.0.1:5001")
run = start_mlflow_run(run_name=run_name, tags=tags)
 

if run:

    import pandas as pd
    import numpy as np
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
        
    # Debugging: Check active run
    print(f"Active run_id: {run.info.run_id}")
 
    # Define n_timesteps and n_features
    n_timesteps = X_train.shape[1]  
    n_features = X_train.shape[2]   
 
    # Model building
    lstm_model = Sequential()
 
    # Add the LSTM layers and dropout layers
    lstm_model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(50, activation='relu'))
    lstm_model.add(Dropout(0.2))
 
    # Output layer
    lstm_model.add(Dense(1))
 
    # Compiling the model with Adam optimizer and MSE loss
    optimizer = Adam(learning_rate=0.001)
    lstm_model.compile(optimizer=optimizer, loss='mse')
 
    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
 
    # Fit the model
    lstm_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
 
    # Making predictions on the test set
    y_test_pred = lstm_model.predict(X_test)
    
 
    # Evaluate model performance
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    
    # Evaluate the model on the test set
    print("LSTM Test Set Metrics:")
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared (R²):", r2)
 
    # Set parameters and log metrics
    log_metric("MSE", mse)
    log_metric("MAE", mae)
    log_metric("R2", r2)

    
    # Debugging: Tracking URI to ensure it's set correctly
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    print("Tracking URL type: ", tracking_url_type_store)
    
    # Infer signature
    predictions_lstm = lstm_model.predict(X_train)
    # signature_lstm = infer_signature(X_train, predictions_lstm)
    
    # # Logging the LSTM model with signature
    # mlflow.keras.log_model(lstm_model, "LSTM model", signature = signature_lstm)
    # Log the model using the new function from mlflow_utils.py
    log_model(lstm_model, "LSTM model", X_train=X_train, predictions=predictions_lstm)

    # Debugging: Confirm run status
    print(f"Run {run.info.run_id} finished successfully!") 

    # End the run
    end_run()

else:
    print("MLflow run was not started. Check for errors.")