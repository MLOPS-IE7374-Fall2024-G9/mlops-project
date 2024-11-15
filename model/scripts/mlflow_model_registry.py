import os
import pickle
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException


class MLflowModelRegistry:
    def __init__(self, tracking_uri: str):
        """
        Initializes the MLflow client and sets the tracking URI.

        Args:
            tracking_uri (str): The URI for the MLflow tracking server.
        """
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register_model(self, model_path: str, model_name: str, run_id: str):
        """
        Registers a model to the MLflow model registry.

        Args:
            model_path (str): The path to the model artifact.
            model_name (str): The name of the model in the registry.
            run_id (str): The ID of the MLflow run associated with the model.
        """
        try:
            result = self.client.create_registered_model(model_name)
            print(f"Model '{model_name}' created in registry.")
        except RestException:
            print(f"Model '{model_name}' already exists in registry.")

        model_uri = f"runs:/{run_id}/{model_path}"
        self.client.create_model_version(
            name=model_name, source=model_uri, run_id=run_id
        )
        print(f"Model version registered: {model_uri}")

    def get_best_model(self, model_name: str, metric: str, ascending: bool = True):
        """
        Retrieves the best model version based on a specific metric.

        Args:
            model_name (str): The name of the model in the registry.
            metric (str): The metric to evaluate the models on.
            ascending (bool): Whether to sort metrics in ascending order (default: True).

        Returns:
            dict: Information about the best model version.
        """
        # Search for all versions of the model in the registry
        model_versions = self.client.search_model_versions(f"name='{model_name}'")

        # Collect run IDs associated with each model version
        run_ids = [version.run_id for version in model_versions]
        print(model_versions)
        print(run_ids)
        print("#########")
        # Query runs associated with these run IDs and filter by the specified metric
        runs = mlflow.search_runs(
            experiment_ids=None,
            filter_string=f"run_id IN {tuple(run_ids)}",
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        )

        # If no runs are found, return None
        if runs.empty:
            print("No model found with the specified metric.")
            return None

        # Get the best run (first row after sorting)
        best_run = runs.iloc[0]

        # Find corresponding model version based on run_id
        best_model_version = next(
            version
            for version in model_versions
            if version.run_id == best_run["run_id"]
        )

        print(
            f"Best model: {best_model_version.version} with {metric}: {best_run[f'metrics.{metric}']}"
        )

        return {
            "version": best_model_version.version,
            "metric": best_run[f"metrics.{metric}"],
        }

    def revert_to_previous_version(self, model_name: str):
        """
        Reverts to the previous version of a model if it exists.

        Args:
            model_name (str): The name of the model in the registry.

        Returns:
            dict: Information about the reverted model version.
        """
        model_versions = self.client.search_model_versions(f"name='{model_name}'")
        if len(model_versions) < 2:
            print("No previous version available to revert to.")
            return None

        latest_version = max(model_versions, key=lambda v: int(v.version))
        previous_version = max(
            [v for v in model_versions if int(v.version) < int(latest_version.version)],
            key=lambda v: int(v.version),
        )
        print(f"Reverted to model version: {previous_version.version}")
        return {"version": previous_version.version, "details": previous_version}

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


            # Fetch the model artifact URI
            model_uri = f"runs:/{run_id}/model"
            print(model_uri)
            # Load the model
            model = mlflow.pyfunc.load_model(model_uri)

            print(
                f"Successfully fetched and loaded the latest model from run ID: {run_id}"
            )
            if 'lr' in run_name:
                model_file_name = 'lr_model_latest.pkl'
            elif 'xgboost' in run_name:
                model_file_name = 'xgboost_model_latest.pkl'
            model_path = os.path.join(os.path.dirname(__file__), f'../pickle/{model_file_name}')
            pickle.dump(model, open(model_path, 'wb'))
            return model, model_file_name

        except Exception as e:
            print(f"Error fetching or initializing the model: {e}")
            return None


# Example usage
if __name__ == "__main__":
    registry = MLflowModelRegistry(tracking_uri="http://34.56.170.84:5000")

    # Register a model
    # registry.register_model(model_path="Linear Regression model", model_name="LR", run_id="69828085f97b48378e5bac8879f635d8")

    # Fetch the best model based on a metric
    best_model = registry.get_best_model("LR", metric="MSE", ascending=False)

    # Revert to the previous version of a model
    # reverted_model = registry.revert_to_previous_version("LR")

    # # Transition a model to production stage
    # registry.transition_model_stage("LR", version=2, stage="Production")

    # List all models
    # models = registry.list_models()
    # print(models)
