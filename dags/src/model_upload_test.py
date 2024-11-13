from google.cloud import storage
from datetime import datetime
import os

# Set the path to your service account key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\misja\OneDrive\Desktop\JAHNAVI\NEU\MLOps\noble-velocity-441519-c9-7d0a38e31972.json"

def upload_model_to_gcs(local_model_path, bucket_name="mlops-g9-bucket", model_name="xgb_model"):
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

# Example usage
local_model_path = r"C:\Users\misja\OneDrive\Desktop\JAHNAVI\NEU\mlops-project\model\pickle\xgboost_model_2024-11-12_23-47-17.pkl"  # Your model's local path
bucket_name = "mlops-g9-bucket"  # Replace with your actual GCS bucket name

# Upload the saved model with a timestamp
model_uri = upload_model_to_gcs(local_model_path, bucket_name)
print(f"Model uploaded to GCS at: {model_uri}")
