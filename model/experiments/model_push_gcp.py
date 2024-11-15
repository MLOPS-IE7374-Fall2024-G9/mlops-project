import shutil
from google.cloud import storage

# Path to your local model file
local_model_path = '/Users/amoghagadde/Desktop/Amogha/Northeastern/SEM_3/ML_Ops/Project/mlops-project/model/pickle/best_model/model.pkl'
gcs_bucket_name = 'mlflow-storage-bucket-mlops-7374'
gcs_model_path = 'mlflow-artifacts/model/best_model.pkl'

# Initialize the GCS client
storage_client = storage.Client()

# Upload the model file to GCS
bucket = storage_client.get_bucket(gcs_bucket_name)
blob = bucket.blob(gcs_model_path)

# Upload the file to GCS
blob.upload_from_filename(local_model_path)
print(f"Model successfully uploaded to gs://{gcs_bucket_name}/{gcs_model_path}")
