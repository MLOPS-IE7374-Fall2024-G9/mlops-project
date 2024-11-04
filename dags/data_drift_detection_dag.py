from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import pandas as pd
from src.data_download import get_data_from_dvc
from src.data_drift_detection import DataDriftDetector 

# ------------------------------------------------------------------------------------------------
# default args
default_args = {
    'owner': 'Group 9',
    'start_date': datetime(2023, 9, 17),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
    "execution_timeout": timedelta(minutes=10),
}

# Data DAG pipeline init
drift_data_dag = DAG(
    "drift_data_dag",
    default_args=default_args,
    description="Data Drift Detection DAG",
    schedule_interval=None,
    catchup=False,
    tags=['drift_data_dag']
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
    dag=drift_data_dag,
    on_failure_callback=email_notify_failure,
    on_success_callback=email_notify_success, 
)

# ------------------------------------------------------------------------------------------------
# Task to load data
def load_data(**kwargs):
    file_path = get_data_from_dvc('data_raw.csv')
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_point = int(len(df) * 0.7)
    baseline_df = df.iloc[:split_point]
    new_data_df = df.iloc[split_point:]
    kwargs['ti'].xcom_push(key='baseline_df', value=baseline_df.to_dict())
    kwargs['ti'].xcom_push(key='new_data_df', value=new_data_df.to_dict())

# Initialize DataDriftDetector and call Evidently drift detection
def detect_drift_evidently(**kwargs):
    baseline_dict = kwargs['ti'].xcom_pull(key='baseline_df', task_ids='load_data')
    new_data_dict = kwargs['ti'].xcom_pull(key='new_data_df', task_ids='load_data')
    baseline_df = pd.DataFrame.from_dict(baseline_dict)
    new_data_df = pd.DataFrame.from_dict(new_data_dict)

    detector = DataDriftDetector(baseline_df, new_data_df)
    evidently_results = detector.detect_drift_evidently()
    print("Evidently Drift Detection Results:", evidently_results)
    kwargs['ti'].xcom_push(key='evidently_results', value=evidently_results)

# Kolmogorov-Smirnov Test for drift detection
def detect_drift_ks_test(**kwargs):
    baseline_dict = kwargs['ti'].xcom_pull(key='baseline_df', task_ids='load_data')
    new_data_dict = kwargs['ti'].xcom_pull(key='new_data_df', task_ids='load_data')
    baseline_df = pd.DataFrame.from_dict(baseline_dict)
    new_data_df = pd.DataFrame.from_dict(new_data_dict)

    detector = DataDriftDetector(baseline_df, new_data_df)
    ks_test_results = detector.detect_drift_ks_test()
    print("KS Test Drift Results:", ks_test_results)
    kwargs['ti'].xcom_push(key='ks_test_results', value=ks_test_results)

# ------------------------------------------------------------------------------------------------
# Define tasks in the DAG
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    provide_context=True,
    dag=drift_data_dag
)

evidently_task = PythonOperator(
    task_id='detect_drift_evidently',
    python_callable=detect_drift_evidently,
    provide_context=True,
    dag=drift_data_dag
)

ks_test_task = PythonOperator(
    task_id='detect_drift_ks_test',
    python_callable=detect_drift_ks_test,
    provide_context=True,
    dag=drift_data_dag
)

# ------------------------------------------------------------------------------------------------
# Set task dependencies
load_data_task >> [evidently_task, ks_test_task] >> send_email
