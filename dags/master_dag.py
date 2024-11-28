from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta

# ------------------------------------------------------------------------------------------------
# default args
default_args = {
    'owner': 'Group 9',
    'start_date': datetime.datetime(2023, 9, 17),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
    "execution_timeout": timedelta(minutes=10),
}

# Data DAG pipeline init
master_dag = DAG(
    "master_dag",
    default_args=default_args,
    description="Dag which operates all other dags",
    schedule_interval='@daily',
    catchup=False,
    tags=['new_data_dag']
)
# ------------------------------------------------------------------------------------------------
# collect new data
trigger_new_data_dag = TriggerDagRunOperator(
    task_id="trigger_new_data_dag",
    trigger_dag_id="new_data_dag",  
    wait_for_completion=True,
)

# data bias mitigation
trigger_data_bias_dag = TriggerDagRunOperator(
    task_id="trigger_data_bias_dag",
    trigger_dag_id="bias_detection_and_mitigation",  
    wait_for_completion=True,
)

# data drift detection
trigger_data_drift_dag = TriggerDagRunOperator(
    task_id="trigger_data_drift_dag",
    trigger_dag_id="drift_data_dag",  
    wait_for_completion=True,
)

# data feature importance analyzer
trigger_feature_imp_dag = TriggerDagRunOperator(
    task_id="trigger_feature_imp_dag",
    trigger_dag_id="feature_imp_analysis_dag",  
    wait_for_completion=True,
)

# model train and evaluate
trigger_model_train_dag = TriggerDagRunOperator(
    task_id="trigger_model_train_dag",
    trigger_dag_id="model_train_evaluate",  
    wait_for_completion=True,
)

# model bias detection and mitigation
trigger_model_bias_dag = TriggerDagRunOperator(
    task_id="trigger_model_bias_dag",
    trigger_dag_id="model_bias_detection_and_mitigation",  
    wait_for_completion=True,
)


trigger_new_data_dag >> trigger_data_bias_dag >> [trigger_data_drift_dag, trigger_feature_imp_dag] >> trigger_model_train_dag >> trigger_model_bias_dag