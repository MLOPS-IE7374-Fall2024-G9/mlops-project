from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime

# Define the DAG
with DAG(
    dag_id="model_retrain_evaluate_dag",
    start_date=datetime(2023, 1, 1),  # Replace with an appropriate start date
    schedule_interval=None,           # Set schedule interval as needed
    catchup=False,                     # Disable backfilling for this example
    tags=['model_retrain_evaluate_dag']
) as dag:

    # collect new data
    trigger_new_data_dag = TriggerDagRunOperator(
        task_id="trigger_new_data_dag",
        trigger_dag_id="new_data_dag",  
        wait_for_completion=True,
    )

    # model train and evaluate
    trigger_model_train_dag = TriggerDagRunOperator(
        task_id="trigger_model_train_dag",
        trigger_dag_id="model_train_evaluate",  
        wait_for_completion=True,
    )

    # Define task dependencies
    trigger_new_data_dag >> trigger_model_train_dag
