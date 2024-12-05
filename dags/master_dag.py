from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta

# pull latest dvc data
# data bias detection -> email
# data feature analysis -> email
# data drift detection -> report gen
# model training -> deployment 
# model bias detection -> rollback + retaining

# ------------------------------------------------------------------------------------------------
# Default arguments
default_args = {
    'owner': 'Group 9',
    'start_date': datetime(2023, 9, 17),
    'retries': 0,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
    "execution_timeout": timedelta(minutes=10),
}

# Data DAG pipeline init
with DAG(
    "master_dag",
    default_args=default_args,
    description="Dag which operates all other dags",
    schedule_interval='@weekly',
    catchup=False,
    tags=['new_data_dag']
) as master_dag:

    # Data bias mitigation
    trigger_data_bias_dag = TriggerDagRunOperator(
        task_id="trigger_data_bias_dag",
        trigger_dag_id="bias_detection_and_mitigation",
        wait_for_completion=True,
    )

    # Data drift detection
    trigger_data_drift_dag = TriggerDagRunOperator(
        task_id="trigger_data_drift_dag",
        trigger_dag_id="drift_data_dag",
        wait_for_completion=True,
    )

    # Data feature importance analyzer
    trigger_feature_imp_dag = TriggerDagRunOperator(
        task_id="trigger_feature_imp_dag",
        trigger_dag_id="feature_imp_analysis_dag",
        wait_for_completion=True,
    )

    # Model train and evaluate
    trigger_model_train_dag = TriggerDagRunOperator(
        task_id="trigger_model_train_dag",
        trigger_dag_id="model_train_evaluate",
        wait_for_completion=True,
    )
    # Model bias detection and mitigation
    trigger_model_bias_dag = TriggerDagRunOperator(
        task_id="trigger_model_bias_dag",
        trigger_dag_id="model_bias_detection_and_mitigation",
        wait_for_completion=True,
    )

    # Define task dependencies
    trigger_data_bias_dag >> [trigger_data_drift_dag, trigger_feature_imp_dag] >> trigger_model_train_dag >> trigger_model_bias_dag
