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
data_feature_imp_analysis_dag = DAG(
    "data_feature_imp_analysis_dag",
    default_args=default_args,
    description="Feature Importance Analysis DAG",
    schedule_interval=None,
    catchup=False,
    tags=['data_feature_imp_analysis_dag']
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
    dag=data_feature_imp_analysis_dag,
    on_failure_callback=email_notify_failure,
    on_success_callback=email_notify_success, 
)

# ------------------------------------------------------------------------------------------------

current_script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_script_dir, '../../model/pickle/xgboost_model.pkl')
data_path = os.path.join(current_script_dir, '../../dataset/data/bias_mitigated_data.csv')

#DAG task

feature_importance_task = PythonOperator(
    task_id='analyze_feature_importance',
    python_callable=analyze_features,
    op_args=[model_path, data_path],  # Pass the model and data paths as arguments
)

# ------------------------------------------------------------------------------------------------
# Task dependencies
feature_importance_task >> send_email