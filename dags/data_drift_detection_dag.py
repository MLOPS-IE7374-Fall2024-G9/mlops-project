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
# Email operator
def email_notify_success(context):
    success_email = EmailOperator(
        task_id='success_email',
        to=['keshri.r@northeastern.edu'],
        subject='Task Success',
        html_content='<p> Data drift dag succeeded.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

def email_notify_failure(context):
    success_email = EmailOperator(
        task_id='failure_email',
        to=['keshri.r@northeastern.edu'],
        subject='Task Failure',
        html_content='<p>Data drift dag Failed.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

send_email = EmailOperator(
    task_id='send_email',
    to='keshri.r@northeastern.edu',    # Email address of the recipient
    subject='Notification from Airflow',
    html_content='<p>This is a notification email sent from Airflow. </p>',
    dag=new_data_dag,
    on_failure_callback=email_notify_failure,
    on_success_callback=email_notify_success, 
)

# ------------------------------------------------------------------------------------------------