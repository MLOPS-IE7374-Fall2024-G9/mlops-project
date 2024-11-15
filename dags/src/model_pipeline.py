import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import subprocess
import pandas as pd
import joblib
from google.cloud import storage
from datetime import datetime
import sys
import os
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from model.scripts.train import *
from model.scripts.validate import *

def upload_model_to_gcs(bucket_name="mlops-g9-bucket", model_name="xgboost"):
    local_directory = os.path.dirname(os.path.abspath(__file__)) + "../../model/pickle/"
    
    if model_name == "xgboost":
        xgboost_files = glob.glob(os.path.join(local_directory, model_name + "_model_*.pkl"))
        if len(xgboost_files) == 0:
            raise("No XGB Model found")
        
        local_model_path = xgboost_files[0]
    else:
        raise("Other models are not present on cloud")
    
    # Initialize Google Cloud Storage client
    storage_client = storage.Client(project="noble-velocity-441519-c9")

    model_filename = os.path.basename(local_model_path)

    # Specify the bucket and upload the model
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(local_model_path)

    model_uri = f"gs://{bucket_name}/models/{model_filename}"
    print(f"Model uploaded to Google Cloud Storage: {model_uri}")
    return model_uri

# Function to Pull Data from DVC and Remove the datetime Column
def load_processed_data(filename="bias_mitigated_data.csv",target_column="value", test_size=0.2, random_state=42):
    try:
        # Pull the file from DVC
        subprocess.run(["dvc", "pull", filename], check=True)
        print(f"{filename} pulled from DVC.")

        # Load the CSV into a DataFrame
        df = pd.read_csv(filename)

        # Remove the datetime column if it exists
        if "datetime" in df.columns:
            df = df.drop(columns=["datetime"])
            print("Removed 'datetime' column from DataFrame.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print(f"Data split into train and test sets with test size {test_size}.")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None


def download_model_from_gcs(model_name, bucket_name="mlops-g9-bucket"):

    if model_name == "xgboost":
        gcs_model_uri = "gs://" + bucket_name + "/models/" + model_name + "_model_.pkl"
    else:
        raise("Other models are not present on cloud")
    
    local_directory = os.path.dirname(os.path.abspath(__file__)) + "../../model/pickle/"

    # Initialize the Google Cloud Storage client
    storage_client = storage.Client()
    
    # Parse the bucket name and blob name from the GCS URI
    bucket_name = gcs_model_uri.split("/")[2]
    blob_name = "/".join(gcs_model_uri.split("/")[3:])

    # Specify the bucket and blob
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Construct the local path with the filename from the GCS URI
    model_filename = os.path.basename(blob_name)
    local_model_path = os.path.join(local_directory, model_filename)

    # Download the model file to the local path
    blob.download_to_filename(local_model_path)
    print(f"Model downloaded from GCS to {local_model_path}")
    return local_model_path

def load_model(gcs_model_uri):
    # Download model from GCS and load it
    local_model_path = download_model_from_gcs(gcs_model_uri)
    model = joblib.load(local_model_path)
    print("Model loaded for predictions.")
    return model

# Function to Test the Model
def test_and_evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Model Test Accuracy: {r2}")
    print(f"Model Mean Squared Error: {mse}")

    return {"r2 score": r2, "mse": mse}


# ---------------------------------------------------------------

def download_model_artifacts():
    local_directory = os.path.dirname(os.path.abspath(__file__)) + "../../model/pickle/"

    return local_directory

def train_model(data_path, model_name, load_existing_model=False):
    trainer = ModelTrainer(load_existing_model=load_existing_model)
    trainer.load_dataset(data_path)
    trainer.train(model_name)
    return model_name

def validate_model(model_finetune, model_train, dataset_path, thresholds):
    trainer = ModelTrainer(load_existing_model=False)
    trainer.load_dataset(dataset_path)

    if model_finetune:
        trainer.load_model_artifact(model_finetune)
    else:
        trainer.load_model_artifact(model_train)
    mse, mae, r2 = trainer.evaluate()

    return mse, mae, r2

def threshold_verification(thresholds, validation_outputs):
    # validate if the model is under the threshold
    mse, mae, r2 = validation_outputs


    
    



