# model_rollback_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import json
import os
from src.model_roll_back import rollback_model  # Import the rollback function from the other file


# default args
default_args = {
    'owner': 'Group 9',
    'start_date': datetime(2023, 9, 17),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
    "execution_timeout": timedelta(minutes=10),
}

# Load the JSON config file from the same directory
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

# Load the JSON config
with open(config_path, 'r') as file:
    config = json.load(file)

# Function to run the rollback task
def run_rollback():
    """Run the rollback for the specified model."""
    # Get model name from JSON config or default to 'model'
    model_name = config.get('mlflow', {}).get('model_name', 'model')
    rollback_model(model_name)
    

# Define the DAG
with DAG(
    'model_rollback',
    default_args=default_args,
    description='DAG to automate model rollback in MLflow',
    schedule=None,  # Set to your desired frequency (e.g., daily)
    catchup=False,  # Don't backfill previous runs
    tags=['model_rollback_dag']
) as dag:

    # Define a task in the DAG
    rollback_task = PythonOperator(
        task_id='run_rollback_task',
        python_callable=run_rollback,
        dag=dag,
    )

    # Set the task dependencies (only one task in this case)
    rollback_task
