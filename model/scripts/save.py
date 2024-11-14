import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import subprocess
import pandas as pd
from google.cloud import storage
from datetime import datetime
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, bucket_name="mlops-g9-bucket", project_id="noble-velocity-441519-c9"):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.storage_client = storage.Client(project=project_id)

    def upload_model_to_gcs(self, local_model_path, model_name="xgboost_model"):
        """Uploads a model to Google Cloud Storage with a unique timestamped filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(local_model_path)

        model_uri = f"gs://{self.bucket_name}/models/{model_filename}"
        logger.info(f"Model uploaded to Google Cloud Storage: {model_uri}")
        return model_uri

    def download_model_from_gcs(self, gcs_model_uri):
        """Downloads a model from Google Cloud Storage to a local directory."""
        local_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../model/pickle/")
        os.makedirs(local_directory, exist_ok=True)

        bucket_name = gcs_model_uri.split("/")[2]
        blob_name = "/".join(gcs_model_uri.split("/")[3:])
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        model_filename = os.path.basename(blob_name)
        local_model_path = os.path.join(local_directory, model_filename)
        blob.download_to_filename(local_model_path)
        
        logger.info(f"Model downloaded from GCS to {local_model_path}")
        return local_model_path

    def load_model(self, gcs_model_uri):
        """Loads a model from GCS for predictions."""
        local_model_path = self.download_model_from_gcs(gcs_model_uri)
        model = joblib.load(local_model_path)
        logger.info("Model loaded for predictions.")
        return model

def main():
    parser = argparse.ArgumentParser(description="Model Manager CLI for uploading and downloading models.")
    parser.add_argument("--upload", action="store_true", help="Upload a model to GCS")
    parser.add_argument("--download", action="store_true", help="Download a model from GCS")
    parser.add_argument("--model-path", type=str, help="Local path to the model for upload/download")
    parser.add_argument("--gcs-model-uri", type=str, help="GCS URI of the model to download")
    parser.add_argument("--bucket-name", type=str, default="mlops-g9-bucket", help="GCS bucket name")
    parser.add_argument("--project-id", type=str, default="noble-velocity-441519-c9", help="Google Cloud project ID")

    args = parser.parse_args()
    manager = ModelManager(bucket_name=args.bucket_name, project_id=args.project_id)

    if args.upload:
        if not args.model_path:
            logger.error("Please specify the --model-path for uploading.")
            return
        model_uri = manager.upload_model_to_gcs(args.model_path)
        logger.info(f"Uploaded model URI: {model_uri}")

    elif args.download:
        if not args.gcs_model_uri:
            logger.error("Please specify the --gcs-model-uri for downloading.")
            return
        local_path = manager.download_model_from_gcs(args.gcs_model_uri)
        logger.info(f"Model downloaded to: {local_path}")

if __name__ == "__main__":
    main()


# usage
# python save.py --upload --model-path /path/to/your/local_model.pkl --bucket-name your-bucket-name --project-id your-project-id
# python script_name.py --download --gcs-model-uri gs://your-bucket-name/models/your_model_filename.pkl --bucket-name your-bucket-name --project-id your-project-id
