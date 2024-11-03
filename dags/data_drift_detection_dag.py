from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import pandas as pd
from src.data_download import get_data_from_dvc
from src.data_drift_detection import DataDriftDetector 

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
new_data_dag = DAG(
    "new_data_dag",
    default_args=default_args,
    description="Data Drift Detection DAG",
    schedule_interval=None,
    catchup=False,
    tags=['new_data_dag']
)

# ------------------------------------------------------------------------------------------------