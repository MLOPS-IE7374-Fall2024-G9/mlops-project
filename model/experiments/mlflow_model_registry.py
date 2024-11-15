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

    def register_model(
        self, run_id: str, model_name: str, metric: str, ascending: bool = True
    ):
        """
        Registers a model as either a champion or a challenger based on its performance.

        Args:
            run_id (str): The run ID of the new model to be registered.
            model_name (str): The name of the model in the registry.
            metric (str): The metric to evaluate models on.
            ascending (bool): Whether lower metric values are better (default: True).

        Returns:
            dict: Information about the registered model and its status (champion or challenger).
        """
        # Get the metric value for the new model
        run = self.client.get_run(run_id)
        new_metric_value = run.data.metrics.get(metric)

        if new_metric_value is None:
            raise ValueError(f"Metric '{metric}' not found in run {run_id}")

        try:
            # Try to retrieve existing versions of the model
            model_versions = self.client.search_model_versions(f"name='{model_name}'")

            # If there are existing models, find the best current version based on the metric
            best_model_version = None
            best_metric_value = None

            for version in model_versions:
                current_run = self.client.get_run(version.run_id)
                current_metric_value = current_run.data.metrics.get(metric)

                if current_metric_value is not None:
                    if (
                        best_model_version is None
                        or (ascending and current_metric_value < best_metric_value)
                        or (not ascending and current_metric_value > best_metric_value)
                    ):
                        best_model_version = version
                        best_metric_value = current_metric_value

            # Compare new model's performance with the best existing model
            if (ascending and new_metric_value < best_metric_value) or (
                not ascending and new_metric_value > best_metric_value
            ):
                # Register new model as champion if it's better
                registered_model = mlflow.register_model(
                    f"runs:/{run_id}/model", model_name
                )

                # Assign alias "champion" to this new version
                mlflow.update_model_version_alias(
                    model_name, registered_model.version, alias="champion"
                )

                print(
                    f"New champion registered: Version {registered_model.version} with {metric}: {new_metric_value}"
                )
                return {
                    "version": registered_model.version,
                    "metric": new_metric_value,
                    "status": "champion",
                }
            else:
                # Register it as a challenger if it's not better
                registered_model = mlflow.register_model(
                    f"runs:/{run_id}/model", model_name
                )

                # Assign alias "challenger" to this version
                mlflow.update_model_version_alias(
                    model_name, registered_model.version, alias="challenger"
                )

                print(
                    f"New challenger registered: Version {registered_model.version} with {metric}: {new_metric_value}"
                )
                return {
                    "version": registered_model.version,
                    "metric": new_metric_value,
                    "status": "challenger",
                }

        except RestException as e:
            # If no models are found in the registry (i.e., first time registering this model), register as champion
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                registered_model = mlflow.register_model(
                    f"runs:/{run_id}/model", model_name
                )

                # Assign alias "champion" since this is the first version
                mlflow.update_model_version_alias(
                    model_name, registered_model.version, alias="champion"
                )

                print(
                    f"First version of '{model_name}' registered as champion: Version {registered_model.version} with {metric}: {new_metric_value}"
                )
                return {
                    "version": registered_model.version,
                    "metric": new_metric_value,
                    "status": "champion",
                }
            else:
                raise e

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
