import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
import mlflow.keras
import mlflow.xgboost
from urllib.parse import urlparse
import logging

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Add a console handler to output logs to console
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Function to set the MLflow tracking URI
def set_tracking_uri(uri):
    mlflow.set_tracking_uri(uri)
    logger.info(f"Set MLflow tracking URI to: {uri}")

def start_mlflow_run(run_name=None, tags=None):
    client = MlflowClient()
    experiment_name = "Electricity Demand Prediction 2.0"
    # experiment_name = "Test 3.0"
    

    try:
        # Check if the experiment exists
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.info(f"Experiment '{experiment_name}' does not exist. Creating a new one.")
            experiment_id = client.create_experiment(experiment_name)
            logger.info(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Found existing experiment '{experiment_name}' with ID: {experiment_id}")

        # Set the experiment
        mlflow.set_experiment(experiment_name)
        logger.info(f"Setting active experiment to '{experiment_name}'.")

        # Start a new MLflow run within the specified experiment
        run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
        logger.info(f"Started run '{run_name}' with ID: {run.info.run_id}")

        # Add tags if provided
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
                logger.info(f"Set tag '{key}' to '{value}'.")

        return run

    except Exception as e:
        logger.error(f"Error in starting MLflow run: {e}")
        raise

# Function to log train data
def log_train_input(X_train):
    try: 
        x_train_log = mlflow.data.from_pandas(X_train, name="Training Dataset")
        mlflow.log_input(x_train_log, context="Train")
        logger.info("Logged train data")
    except Exception as e:
        logger.error("Error logging train data")
    
# Function to log Evaluation data
def log_val_input(X_val):
    try: 
        x_val_log = mlflow.data.from_pandas(X_val, name="Evaluation Dataset")
        mlflow.log_input(x_val_log, context="Evaluation")
        logger.info("Logged Evaluation data")
    except Exception as e:
        logger.error("Error logging Evaluation data")

# Function to log a metric
def log_metric(metric_name, value, step=None):
    try:
        mlflow.log_metric(metric_name, value, step)
        logger.info(f"Logged metric '{metric_name}': {value}")
    except Exception as e:
        logger.error(f"Error logging metric '{metric_name}': {e}")

# Function to log a parameter
def log_param(param_name, value):
    try:
        mlflow.log_param(param_name, value)
        logger.info(f"Logged parameter '{param_name}': {value}")
    except Exception as e:
        logger.error(f"Error logging parameter '{param_name}': {e}")

# Function to log a model
def log_model(model, model_name, X_train=None, predictions=None, signature=None):
    try:
        # Infer the signature only if X_train and predictions are provided
        if signature is None and X_train is not None and predictions is not None:
            signature = infer_signature(X_train, predictions)
        
        if model_name == "xgboost":
            mlflow.xgboost.log_model(model, "xgboost", signature=signature)
        elif model_name == "lr":
            mlflow.sklearn.log_model(model, "lr", signature=signature)
        else:
            mlflow.tensorflow.log_model(model, "lstm", signature=signature)
        logger.info(f"Logged model '{model_name}' with signature.")
    except Exception as e:
        logger.error(f"Error logging model '{model_name}': {e}")

# Function to end the MLflow run
def end_run():
    try:
        mlflow.end_run()
        logger.info("Ended the MLflow run.")
    except Exception as e:
        logger.error(f"Error ending the MLflow run: {e}")
