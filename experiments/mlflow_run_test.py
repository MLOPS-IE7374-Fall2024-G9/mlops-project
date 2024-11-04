import mlflow

# Replace 'experiment_id' with the ID of your experiment
experiment_id = "1"
active_runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string="status = 'RUNNING'")

# Display active runs
print(active_runs)
