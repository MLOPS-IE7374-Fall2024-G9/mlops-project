# from mlflow.tracking import MlflowClient

# def download_artifacts(run_id, artifact_path, destination_path):
#     client = MlflowClient()
#     # Download artifact directory from MLflow to a local path
#     client.download_artifacts(run_id=run_id, path=artifact_path, dst_path=destination_path)

# # Usage
# download_artifacts(
#     run_id="abfc86e7ed604615bea3a6fd30562df2",
#     artifact_path="",  # Leaving empty to download all artifacts
#     destination_path="./model/pickel/save_artifacts_gcp/"
# )


from google.cloud import storage
import os

def download_gcs_artifacts(bucket_name, artifact_path, destination_path):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=artifact_path)

    for blob in blobs:
        # Define destination path
        destination_file_path = f"{destination_path}/{blob.name[len(artifact_path):]}"
        os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
        blob.download_to_filename(destination_file_path)
        print(f"Downloaded {blob.name} to {destination_file_path}")

# Usage
download_gcs_artifacts(
    bucket_name="mlflow-storage-bucket-mlops-7374",
    artifact_path="mlruns/abfc86e7ed604615bea3a6fd30562df2",
    destination_path="./model/pickle/save_artifacts_gcp"
)
