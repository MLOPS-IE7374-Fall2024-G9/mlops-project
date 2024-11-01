import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
 
# Setting MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")
 
# Experiment name
# mlflow.create_experiment("Electricity Demand Prediction_1")
# mlflow.set_experiment("Electricity Demand Prediction_1")
# mlflow.create_experiment("Electricity Demand Prediction_1", artifact_location="/Users/amoghagadde/Desktop/Amogha/Northeastern/SEM_3/ML_Ops/Project/mlops-project/experiments")

experiment = mlflow.get_experiment_by_name("Electricity Demand Prediction_1")
mlflow.set_experiment("Electricity Demand Prediction_1")

print("Current artifact location:", experiment.artifact_location)
 
data = pd.read_csv('/Users/akm/Desktop/mlops-project/preprocessed_data.csv')
X = data.drop(columns=['value'])
y = data[['value']]

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2) 

# Linear Regression Tracking
with mlflow.start_run(run_name="Linear Regression") as run:
    
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
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    #mlflow.sklearn.log_model(lin_reg, "Linear Regression model")
    
    # Debugging: Tracking URI to ensure it's set correctly
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    print("Tracking URL type: ", tracking_url_type_store)
     
    # Infer signature
    predictions_lin = lin_reg.predict(X_train)
    signature_lin = infer_signature(X_train, predictions_lin)
 
    # Logging the model with signature
    mlflow.sklearn.log_model(lin_reg, "Linear Regression model", signature=signature_lin)
    
    # Debugging: Confirm run status
    print(f"Run {run.info.run_id} finished successfully!")
    
 