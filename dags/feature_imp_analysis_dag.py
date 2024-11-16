from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import pandas as pd
import sys
import os
import shap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


from src.feature_analyzer import *

# ------------------------------------------------------------------------------------------------
# default args
default_args = {
    'owner': 'Group 9',
    'start_date': datetime(2023, 9, 17),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
    "execution_timeout": timedelta(minutes=10),
}

# Feature Analsis DAG pipeline init
drift_data_dag = DAG(
    "feature_imp_analysis_dag",
    default_args=default_args,
    description="Feature Importance Analysis DAG",
    schedule_interval=timedelta(days=7),
    catchup=False,
    tags=['feature_imp_analysis_dag']
)


# ------------------------------------------------------------------------------------------------
# Email operator
def email_notify_success(context):
    success_email = EmailOperator(
        task_id='success_email',
        to=["mlops.group.9@gmail.com"],
        subject='Task Success',
        html_content='<p> Feature Analysis dag succeeded.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

def email_notify_failure(context):
    success_email = EmailOperator(
        task_id='failure_email',
        to=["mlops.group.9@gmail.com"],
        subject='Task Failure',
        html_content='<p>Feature Analysis dag Failed.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

send_email = EmailOperator(
    task_id='send_email',
    to=["mlops.group.9@gmail.com"],
    subject='Notification from Airflow',
    html_content='<p>This is a notification email sent from Airflow. </p>',
    dag=drift_data_dag,
    on_failure_callback=email_notify_failure,
    on_success_callback=email_notify_success, 
)

# ------------------------------------------------------------------------------------------------

