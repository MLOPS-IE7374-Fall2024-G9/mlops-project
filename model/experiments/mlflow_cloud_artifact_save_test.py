import mlflow
experiment_name = "test"
mlflow.create_experiment(experiment_name, artifact_location="gs://mlflow-storage-bucket-mlops-7374/mlruns/")
mlflow.set_experiment(experiment_name)
mlflow.set_tracking_uri('http://127.0.0.1:5000')
with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_artifact("/Users/amoghagadde/Desktop/Amogha/Northeastern/SEM_3/ML_Ops/Project/mlops-project/model/pickle/best_model/model.pkl")