from mlflow.tracking import MlflowClient
import mlflow

# Set the tracking URI for a specific server
mlflow.set_tracking_uri("http://127.0.0.1:5001")  # Replace with your server's URL

# Initialize the MLflow client
client = MlflowClient()

def get_current_production_model(model_name):
    """Fetch the latest model in the 'Production' stage."""
    versions = client.search_model_versions(f"name='{model_name}'")
    for version in versions:
        if version.current_stage == 'Production':
            return version
    print(f"No production version found for model: {model_name}")
    return None

def get_previous_model(model_name, current_version):
    """Fetch the model version numerically before the current production version."""
    all_versions = client.search_model_versions(f"name='{model_name}'")
    
    # Sort all versions by version number
    sorted_versions = sorted(all_versions, key=lambda x: int(x.version))
    
    # Find the previous version
    for idx, version in enumerate(sorted_versions):
        if int(version.version) == int(current_version):
            return sorted_versions[idx - 1] if idx > 0 else None
    return None

def rollback_model(model_name):
    """Rollback the current production model to the previous version."""
    # Get the current production model
    current_model = get_current_production_model(model_name)
    if not current_model:
        print("No current production model to replace.")
        return

    # Get the previous model
    previous_model = get_previous_model(model_name, current_model.version)
    if not previous_model:
        print("No previous model version available for rollback.")
        return

    # Transition the previous model to Production
    client.transition_model_version_stage(
        name=model_name,
        version=previous_model.version,
        stage="Production"
    )
    print(f"Version {previous_model.version} has been promoted to Production.")

    # Transition the current model out of Production (e.g., to Archived)
    client.transition_model_version_stage(
        name=model_name,
        version=current_model.version,
        stage="Archived"  # Or "Staging"
    )
    print(f"Version {current_model.version} has been demoted from Production.")


if __name__ == "__main__":
    model_name = "Linear_Regression"  # Replace with the name of your model
    rollback_model(model_name)
