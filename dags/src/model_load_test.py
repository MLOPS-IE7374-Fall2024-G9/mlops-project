import mlflow
import os



# Set the path to your service account key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\misja\OneDrive\Desktop\JAHNAVI\NEU\MLOps\mlops-7374-39acb5ae6a01.json"

# Define the load_model function
def load_model(model_name, stage="Staging"):
    # Load the model from the MLflow Model Registry
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model '{model_name}' loaded from MLflow (Stage: {stage}).")
    return model

# Test parameters
model_name = "XGBoost_v1.0_20241113_181938"  # Replace with your actual model name in the MLflow registry
stage = "Staging"  # or "Staging", "Archived", etc., based on your model's stage in MLflow

# Test the load_model function
try:
    model = load_model(model_name, stage)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")




