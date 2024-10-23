from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime


## test19 dag change overwrite on vm
# Define a simple function for the PythonOperator
def print_hello():
    return "Hello, Airflow! Test 2"


# Define the default arguments for the DAG
default_args = {
    "owner": "airflow",
    "start_date": datetime(2023, 1, 1),  # Adjust the start date as needed
    "retries": 1,
}

# Initialize the DAG
with DAG(
    dag_id="test_airflow_dag1",  # DAG ID
    default_args=default_args,  # Default arguments
    schedule_interval=None,  # Do not schedule automatically, can be triggered manually
    catchup=False,  # Prevent backfilling
) as dag:

    # Dummy start task
    start_task = DummyOperator(task_id="start")

    # Python task that prints "Hello, Airflow!"
    hello_task = PythonOperator(task_id="print_hello", python_callable=print_hello)

    # Dummy end task
    end_task = DummyOperator(task_id="end")

    # Define task dependencies
    start_task >> hello_task >> end_task
