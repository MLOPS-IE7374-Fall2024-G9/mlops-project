from mlflow.tracking import MlflowClient
import os
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MLflow client
client = MlflowClient()


def get_current_production_model(model_name):
    """Fetch the latest model in the 'Production' stage."""
    logging.info(f"Fetching the current production model for: {model_name}")
    versions = client.search_model_versions(f"name='{model_name}'")
    for version in versions:
        if version.current_stage == 'Production':
            logging.info(f"Found production version: {version.version}")
            return version
    logging.warning(f"No production version found for model: {model_name}")
    return None

def get_previous_model(model_name, current_version):
    """Fetch the model version numerically before the current production version."""
    logger.info(f"Fetching previous model version for {model_name}, current version: {current_version}")
    all_versions = client.search_model_versions(f"name='{model_name}'")
    
    # Sort all versions by version number
    sorted_versions = sorted(all_versions, key=lambda x: int(x.version))
    
    # Find the previous version
    for idx, version in enumerate(sorted_versions):
        if int(version.version) == int(current_version):
            if idx > 0:
                logger.info(f"Previous model version found: {sorted_versions[idx - 1].version}")
                return sorted_versions[idx - 1]
            else:
                logger.warning(f"No previous model version available for {model_name} (first version).")
                return None
    logger.warning(f"Model version {current_version} not found for {model_name}.")
    return None

def rollback_model(model_name):
    """Rollback the current production model to the previous version."""
    logger.info(f"Starting rollback for model: {model_name}")
    
    # Get the current production model
    current_model = get_current_production_model(model_name)
    if not current_model:
        logger.error("No current production model to replace.")
        return

    # Get the previous model
    previous_model = get_previous_model(model_name, current_model.version)
    if not previous_model:
        logger.error("No previous model version available for rollback.")
        return

    # Transition the previous model to Production
    logger.info(f"Promoting version {previous_model.version} to Production.")
    client.transition_model_version_stage(
        name=model_name,
        version=previous_model.version,
        stage="Production"
    )
    logger.info(f"Version {previous_model.version} has been promoted to Production.")

    # Transition the current model out of Production (e.g., to Archived)
    logger.info(f"Demoting version {current_model.version} from Production to Archived.")
    client.transition_model_version_stage(
        name=model_name,
        version=current_model.version,
        stage="Archived"  # Or "Staging"
    )
    logger.info(f"Version {current_model.version} has been demoted from Production.")