from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

data= pd.read_csv('/Users/akm/Desktop/mlops-project/preprocessed_data.csv')
X = data.drop(columns=['value'])
y = data[['value']]

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2)
 
X_train = X_train.to_numpy().reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.to_numpy().reshape((X_test.shape[0], 1, X_test.shape[1]))
 
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import mlflow.keras
from mlflow.models import infer_signature
from urllib.parse import urlparse


mlflow.set_tracking_uri("http://localhost:5001")
 
# Experiment name
# mlflow.create_experiment("Electricity Demand Prediction_1")
# mlflow.set_experiment("Electricity Demand Prediction_1")
# mlflow.create_experiment("Electricity Demand Prediction_1", artifact_location="/Users/amoghagadde/Desktop/Amogha/Northeastern/SEM_3/ML_Ops/Project/mlops-project/experiments")

experiment = mlflow.get_experiment_by_name("Electricity Demand Prediction_1")
if experiment is None:
    mlflow.create_experiment("Electricity Demand Prediction_1")
mlflow.set_experiment("Electricity Demand Prediction_1")

mlflow.set_experiment("Electricity Demand Prediction_1")

print("Current artifact location:", experiment.artifact_location)
 
with mlflow.start_run(run_name = "LSTM") as run:
    
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
 
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R-squared (R²): {r2}')
    
    #Logging parameters, metrics, and model
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    
    # Debugging: Tracking URI to ensure it's set correctly
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    print("Tracking URL type: ", tracking_url_type_store)
    
    # Infer signature
    predictions_lstm = lstm_model.predict(X_train)
    signature_lstm = infer_signature(X_train, predictions_lstm)
    
    # Logging the model with signature
    mlflow.keras.log_model(lstm_model, "LSTM model", signature = signature_lstm)
    
    # Debugging: Confirm run status
    print(f"Run {run.info.run_id} finished successfully!")