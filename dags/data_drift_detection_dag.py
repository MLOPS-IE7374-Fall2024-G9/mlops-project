from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import pandas as pd
from src.data_drift import *

# ------------------------------------------------------------------------------------------------
# default args
default_args = {
    'owner': 'Group 9',
    'start_date': datetime(2023, 9, 17),
    'retries': 2, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
    "execution_timeout": timedelta(minutes=10),
}

# Data DAG pipeline init
data_drift_detection_dag = DAG(
    "data_drift_detection_dag",
    default_args=default_args,
    description="Data Drift Detection DAG",
    schedule_interval=None,
    catchup=False,
    tags=['data_drift_detection_dag']
)

filename = "data_preprocess.csv"
drift_report = "data_drift.html"

# ------------------------------------------------------------------------------------------------
# Email operator
def email_notify_success(context):
    success_email = EmailOperator(
        task_id='success_email',
        to=["mlops.group.9@gmail.com"],
        subject='Task Success',
        html_content='<p> Data drift dag succeeded.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

def email_notify_failure(context):
    success_email = EmailOperator(
        task_id='failure_email',
        to=["mlops.group.9@gmail.com"],
        subject='Task Failure',
        html_content='<p>Data drift dag Failed.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

send_email = EmailOperator(
    task_id='send_email',
    to=["mlops.group.9@gmail.com"],
    subject='Notification from Airflow',
    html_content='<p>This is a notification email sent from Airflow. </p>',
    dag=data_drift_detection_dag,
    on_failure_callback=email_notify_failure,
    on_success_callback=email_notify_success, 
)

# ------------------------------------------------------------------------------------------------
# Define tasks in the DAG
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    provide_context=True,
    op_args=[filename],
    dag=data_drift_detection_dag
)

evidently_task = PythonOperator(
    task_id='detect_drift_evidently',
    python_callable=detect_drift_evidently,
    provide_context=True,
    op_args=[drift_report],
    dag=data_drift_detection_dag
)

ks_test_task = PythonOperator(
    task_id='detect_drift_ks_test',
    python_callable=detect_drift_ks_test,
    provide_context=True,
    dag=data_drift_detection_dag
)

psi_task = PythonOperator(
    task_id='detect_drift_psi',
    python_callable=detect_drift_psi,
    provide_context=True,
    dag=data_drift_detection_dag
)

# ------------------------------------------------------------------------------------------------
# Task dependencies
load_data_task >> [evidently_task, ks_test_task, psi_task] >> send_email