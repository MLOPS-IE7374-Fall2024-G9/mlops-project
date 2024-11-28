from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

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

trigger_new_data_dag >> trigger_model_train_dag