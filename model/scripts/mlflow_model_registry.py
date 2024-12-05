import os
import pickle
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
import logging
import argparse
import json

logger = logging.getLogger("ModelRegistry")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class MLflowModelRegistry:
    def __init__(self, tracking_uri: str):
        """
        Initializes the MLflow client and sets the tracking URI.

        Args:
            tracking_uri (str): The URI for the MLflow tracking server.
        """
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.configure_mlfow_credentials("mlops-7374-3e7424e80d76.json")

    def configure_mlfow_credentials(self, json_credential_path):
        """
        Function to run the mflow credentials
        """
        try:
            # Run the dvc remote modify command
            logger.info(f"Configuring mlflow credentials from {json_credential_path}.")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=json_credential_path
            print(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
            logger.info("MLflow remote configuration successful.")

        except Exception as e:
            logger.error(f"Failed to configure mflow remote: {e}")

    # def register_model(self, model_path: str, model_name: str, run_id: str):
    #     """
    #     Registers a model to the MLflow model registry.

    #     Args:
    #         model_path (str): The path to the model artifact.
    #         model_name (str): The name of the model in the registry.
    #         run_id (str): The ID of the MLflow run associated with the model.
    #     """
    #     try:
    #         result = self.client.create_registered_model(model_name)
    #         print(f"Model '{model_name}' created in registry.")
    #     except RestException:
    #         print(f"Model '{model_name}' already exists in registry.")

    #     model_uri = f"runs:/{run_id}/{model_path}"
    #     self.client.create_model_version(
    #         name=model_name, source=model_uri, run_id=run_id
    #     )
    #     print(f"Model version registered: {model_uri}")
    
    def register_model(self, model_name: str, model_uri: str, metrics: dict):
        """
        Registers a new model and manages model stages in the MLflow Model Registry.

        Args:
            model_name (str): Name of the model to register.
            model_uri (str): URI of the model to register.
            metrics (dict): Metrics of the new model (e.g., {'r2': 0.92, 'mse': 0.008}).
        """
        try:
            # Step 1: Check if the required metrics are present
            if metrics.get('R2') is None or metrics.get('MSE') is None:
                print(f"Warning: Some metrics are missing for model '{model_name}'. Skipping registration.")
                return  # Skip registration if required metrics are missing

            # Step 2: Register the new model if metrics are complete
            registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
            new_model_version = registered_model.version
            print(f"Registered new model: {model_name} version {new_model_version}")

            # Step 3: Manage stages and promote the model if metrics are valid
            self.manage_model_stages(model_name, new_model_version, metrics)

        except Exception as e:
            print(f"Error in registering and managing model: {e}")
            
            
    def manage_model_stages(self, model_name: str, model_version: int, metrics: dict):
        """
        Manages the model stages, promoting the best model to production.

        Args:
            model_name (str): The model's name.
            model_version (int): The version of the model to manage.
            metrics (dict): The metrics of the model.
        """
        try:
            # Step 1: Fetch all registered models for the given model name
            model_versions = self.client.search_model_versions(f"name='{model_name}'")

            # Step 2: Check if any model is in the "Production" stage
            models_in_production = [
                model for model in model_versions if model.current_stage == 'Production'
            ]

            # Scenario 1: No model in production
            if not models_in_production:
                # Promote the present model to production
                print(f"No model in production. Promoting model version {model_version} to Production.")
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=model_version,
                    stage='Production',
                    archive_existing_versions=True
                )
                
            
            # Scenario 2: A model is already in production
            else:
                # There's exactly one model in production (at any time) - getting the first one
                current_prod_model = models_in_production[0]
                
                # Fetching the run associated with the current production model
                prod_run = self.client.get_run(current_prod_model.run_id)
                current_prod_metrics = prod_run.data.metrics
                print('Current Model:', current_prod_model.version, current_prod_metrics)

                # Comparing the 'R2' of the current production model and the new model
                new_r2 = metrics.get('R2', float('-inf'))
                current_r2 = current_prod_metrics.get('R2', float('-inf'))

                if new_r2 > current_r2:
                    print(f"New model version {model_version} is better than the current production model version {current_prod_model.version}.")
                    
                    # Promote the new model to production
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=model_version,
                        stage='Production',
                        archive_existing_versions=True
                    )
                    print("Promoting the new model to Production")
                else:
                    print(f"Current production model version {current_prod_model.version} is better or equal to the new model version {model_version}. No promotion to production.")
        except Exception as e:
            print(f"Error managing model stages: {e}")



    def revert_to_previous_version(self, model_name: str):
        """
        Reverts to the previous version of a model if it exists.

        Args:
            model_name (str): The name of the model in the registry.

        Returns:
            dict: Information about the reverted model version.
        """
        model_versions = self.client.search_model_versions(f"name='{model_name}'")
        # if len(model_versions) < 2:
        #     print("No previous version available to revert to.")
        #     return None

        # latest_version = max(model_versions, key=lambda v: int(v.version))
        # previous_version = max(
        #     [v for v in model_versions if int(v.version) < int(latest_version.version)],
        #     key=lambda v: int(v.version),
        # )
        # print(f"Reverted to model version: {previous_version.version}")
        # return {"version": previous_version.version, "details": previous_version}
        
        # Filter for versions that are archived ---- as previous models will be archieved once the new model is found to be better performing
        archived_versions = [v for v in model_versions if v.current_stage == 'Archived']
        
        if not archived_versions:
            print("No archived version available to revert to.")
            return None

        # Find the latest archived version
        latest_archived_version = max(archived_versions, key=lambda v: int(v.version))
        
        print(f"Reverted to archived model version: {latest_archived_version.version}")
        return {"version": latest_archived_version.version, "details": latest_archived_version}

    def transition_model_stage(self, model_name: str, version: int, stage: str):
        """
        Transitions a model to a new stage (e.g., 'Production', 'Staging').

        Args:
            model_name (str): The name of the model in the registry.
            version (int): The version of the model to transition.
            stage (str): The target stage (e.g., 'Production', 'Staging').
        """
        self.client.transition_model_version_stage(
            name=model_name, version=version, stage=stage
        )
        print(f"Model '{model_name}' version {version} transitioned to stage: {stage}")

    def list_models(self):
        """
        Lists all registered models.

        Returns:
            list: A list of registered models and their versions.
        """
        models = self.client.search_registered_models()

        return models

    def fetch_and_initialize_latest_model(self, experiment_name):
        """
        Fetches the latest model trained in an MLflow experiment and initializes it.

        Args:
            experiment_name (str): The name of the MLflow experiment.

        Returns:
            model: The initialized model object.
            path: The local path to model
        """
        try:

            # Get the experiment details
            experiment = self.client.get_experiment_by_name(experiment_name)
            if not experiment:
                raise ValueError(f"Experiment '{experiment_name}' does not exist.")

            # Get the latest run ID from the experiment
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.accuracy DESC"],  # Adjust metric if needed
                max_results=1,
            )

            if not runs:
                raise ValueError(f"No runs found in experiment '{experiment_name}'.")

            latest_run = runs[0]
            run_id = latest_run.info.run_id
            run_name = latest_run.info.run_name
            model_name = run_name.split("_")[0]

            # Fetch the model artifact URI
            model_uri = f"runs:/{run_id}/{model_name}"
            print(model_uri)
            # Load the model
            model = mlflow.pyfunc.load_model(model_uri)

            print(
                f"Successfully fetched and loaded the latest model from run ID: {run_id}"
            )

            model_file_name = None
            model_type = None
            print('Run name: ', run_name)
            if 'lr' in run_name:
                model_file_name = 'lr_model_latest.pkl'
                model_type = "lr"
            elif 'xgboost' in run_name:
                model_file_name = 'xgboost_model_latest.pkl'
                model_type = "xgboost"
            elif 'lstm' in run_name: 
                model_file_name = 'lstm_model_latest.pkl'
                model_type = "lstm"
            model_path = os.path.join(os.path.dirname(__file__), f'../pickle/{model_file_name}')
            pickle.dump(model, open(model_path, 'wb'))
            pickle.dump(model, open(os.path.join(os.path.dirname(__file__), f'../pickle/model.pkl'), 'wb'))
            return model, model_type

        except Exception as e:
            print(f"Error fetching or initializing the model: {e}")
            return None
        
    def get_current_production_model(self, model_name):
        """Fetch the latest model in the 'Production' stage."""
        versions = self.client.search_model_versions(f"name='{model_name}'")
        for version in versions:
            if version.current_stage == 'Production':
                return version
        print(f"No production version found for model: {model_name}")
        return None
    
    def get_previous_model(self, model_name, current_version):
        """Fetch the model version numerically before the current production version."""
        all_versions = self.client.search_model_versions(f"name='{model_name}'")
        
        # Sort all versions by version number
        sorted_versions = sorted(all_versions, key=lambda x: int(x.version))
        
        # Find the previous version
        for idx, version in enumerate(sorted_versions):
            if int(version.version) == int(current_version):
                return sorted_versions[idx - 1] if idx > 0 else None
        return None
    
    def rollback_model(self, model_name):
        """Rollback the current production model to the previous version."""
        # Get the current production model
        current_model = self.get_current_production_model(model_name)
        if not current_model:
            print("No current production model to replace.")
            return
        # Get the previous model
        previous_model = self.get_previous_model(model_name, current_model.version)
        if not previous_model:
            print("No previous model version available for rollback.")
            return

        # Transition the previous model to Production
        self.client.transition_model_version_stage(
            name=model_name,
            version=previous_model.version,
            stage="Production"
        )
        print(f"Version {previous_model.version} has been promoted to Production.")

        # Transition the current model out of Production (e.g., to Archived)
        self.client.transition_model_version_stage(
            name=model_name,
            version=current_model.version,
            stage="Archived"  # Or "Staging"
        )
        print(f"Version {current_model.version} has been demoted from Production.")
    

def test(registry):
    # Register a model
    registry.register_model(model_path="XGBoost model", model_name="model", run_id="35fa103c3b9f44bb80cfd61d478307a8")

    # Fetch the best model based on a metric
    best_model = registry.get_best_model("model", metric="MSE", ascending=False)

    # Revert to the previous version of a model
    reverted_model = registry.revert_to_previous_version("model")

    # # Transition a model to production stage
    # registry.transition_model_stage("LR", version=2, stage="Production")

    # List all models
    models = registry.list_models()
    print(models)

if __name__ == "__main__":

    try:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.json'))
        with open(path, "r") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        print("Error: config.json not found in the current directory.")
        exit(1)
    except json.JSONDecodeError:
        print("Error: Failed to parse config.json. Ensure it is valid JSON.")
        exit(1)
    
    TRACKING_URI = config.get("mlflow_tracking_uri")
    EXPERIMENT_NAME = config.get("experimentation_name")

    if not TRACKING_URI or not EXPERIMENT_NAME:
        print("Error: Missing required configuration in config.json.")
        exit(1)

    registry = MLflowModelRegistry(tracking_uri=TRACKING_URI)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="MLflow Model Registry Operations")
    parser.add_argument(
        "--operation",
        type=str,
        choices=["fetch_latest", "rollback"],
        required=True,
        help="Choose the operation to perform: 'fetch_latest' or 'rollback'.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="Name of the model to operate on.",
    )
    args = parser.parse_args()

    if args.operation == "fetch_latest":
        print(f"Fetching and initializing the latest model from experiment: {EXPERIMENT_NAME}")
        model, model_type = registry.fetch_and_initialize_latest_model(EXPERIMENT_NAME)
        if model:
            print(f"Successfully fetched the latest model of type '{model_type}'.")
        else:
            print("Failed to fetch the latest model.")

    elif args.operation == "rollback":
        print(f"Rolling back the current production model for: {args.model_name}")
        registry.rollback_model(args.model_name)
    
    elif args.operation == "test":
        test()

# usage 
# python mlflow_model_registry.py --operation fetch_latest
# python mlflow_model_registry.py --operation rollback --model_name xgboost

