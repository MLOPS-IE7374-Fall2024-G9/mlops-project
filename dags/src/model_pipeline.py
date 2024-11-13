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


def upload_model_to_gcs(local_model_path, bucket_name="mlops-g9-bucket", model_name="retrained_xgb_model"):
    # Initialize Google Cloud Storage client
    storage_client = storage.Client(project="noble-velocity-441519-c9")

    # Generate a unique filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}.pkl"

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


def download_model_from_gcs(gcs_model_uri, local_directory=r"C:\Users\misja\OneDrive\Desktop\JAHNAVI\NEU\mlops-project\model"):
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


def train_model(df, target_column="value"):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize XGBoost Regressor
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42,reg_alpha=0.1, reg_lambda=1.0,learning_rate=0.05, n_estimators=1000, min_child_weight=5)

    # Train the model
    xgb_reg.fit(X_train, y_train)

    # Save the trained model locally
    # Generate a timestamped filename for the local model path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"xgb_retrained_{timestamp}.pkl"
    model_directory = r"C:\Users\misja\OneDrive\Desktop\JAHNAVI\NEU\mlops-project\model"
    local_model_path = os.path.join(model_directory, model_filename)

    joblib.dump(xgb_reg, local_model_path)
    print(f"Model saved locally at {local_model_path}")

    # Upload the model to Google Cloud Storage
    model_uri = upload_model_to_gcs(local_model_path=local_model_path, bucket_name=bucket_name)
    
    return xgb_reg, X_train, X_test, y_train, y_test, model_uri
    
    


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


# Function to Validate the Model
def validate_model(model, X_train, y_train):
    # Perform validation on training data
    y_pred_train = model.predict(X_train)

    # Calculate performance metrics
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    print(f"Validation MSE: {mse_train}")
    print(f"Validation R2: {r2_train}")
    return mse_train, r2_train


