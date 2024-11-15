import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
import mlflow.sklearn
import mlflow.keras
import mlflow.xgboost
from urllib.parse import urlparse



# Function to set the MLflow tracking URI
def set_tracking_uri(uri):
    mlflow.set_tracking_uri(uri)

def start_mlflow_run(run_name=None, tags=None):
    client = MlflowClient()
    experiment_name = "Electricity Demand Prediction 2.0"

    try:
        # Check if the experiment exists
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' does not exist. Creating a new one.")
            experiment_id = client.create_experiment(experiment_name, artifact_location="gs://mlflow-storage-bucket-mlops-7374/mlruns/")
            print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            print(f"Found existing experiment '{experiment_name}' with ID: {experiment_id}")

        # Set the experiment
        # mlflow.create_experiment(experiment_name, artifact_location="gs://mlflow-storage-bucket-mlops-7374/mlruns/")
        mlflow.set_experiment(experiment_name)
        print(f"Setting active experiment to '{experiment_name}'.")
        
        # # Start the run
        # run = mlflow.start_run(run_name=run_name)
        # print(f"Started run '{run_name}' with ID: {run.info.run_id}")
        
        # Start a new MLflow run within the specified experiment
        run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
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

# Function to log a model
def log_model(model, model_name, X_train=None, predictions=None, signature=None):
    try:
        # Infer the signature only if X_train and predictions are provided
        if signature is None and X_train is not None and predictions is not None:
            signature = infer_signature(X_train, predictions)
        
        mlflow.sklearn.log_model(model, model_name, signature=signature)
        print(f"Logged model '{model_name}' with signature.")
    except Exception as e:
        print(f"Error logging model '{model_name}': {e}")

# Function to end the MLflow run
def end_run():
    try:
        mlflow.end_run()
        print("Ended the MLflow run.")
    except Exception as e:
        print(f"Error ending the MLflow run: {e}")
