from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
from src.data_download import *
from src.data_preprocess import *

# default args
default_args = {
    'owner': 'Group 9',
    'start_date': datetime.datetime(2023, 9, 17),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
    "execution_timeout": timedelta(minutes=10),
}

# Data DAG pipeline init
data_raw_preprocess_dag = DAG(
    "data_raw_preprocess_dag",
    default_args=default_args,
    description="Raw Data Preprocess DAG",
    schedule_interval=None,
    catchup=False,
    tags=['data_raw_preprocess_dag']
)

# ------------------------------------------------------------------------------------------------
# variables
filename_raw = "data_raw.csv"

# ------------------------------------------------------------------------------------------------
# Email operators
def email_notify_success(context):
    success_email = EmailOperator(
        task_id='success_email',
        to=["mlops.group.9@gmail.com"],
        subject='Task Success',
        html_content='<p>Raw data preprocess succeeded.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

def email_notify_failure(context):
    success_email = EmailOperator(
        task_id='failure_email',
        to=["mlops.group.9@gmail.com"],
        subject='Task Failure',
        html_content='<p>Raw data preprocess task Failed.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)
    
send_email = EmailOperator(
    task_id='send_email',
    to=["mlops.group.9@gmail.com"],    # Email address of the recipient
    subject='Notification from Airflow',
    html_content='<p>This is a notification email sent from Airflow. </p>',
    dag=data_raw_preprocess_dag,
    on_failure_callback=email_notify_failure,
    on_success_callback=email_notify_success,
)

# ------------------------------------------------------------------------------------------------
# Python operators
# --------------------------
# Data API Operators
# function to pull preprocessed data from dvc, returns json
data_from_dvc_task = PythonOperator(
    task_id = 'data_from_dvc_task',
    python_callable=get_data_from_dvc,
    provide_context=True,
    op_args=[filename_raw],
    dag = data_raw_preprocess_dag
)

# function to apply all preprocessing steps to raw data
preprocess_pipeline_task = PythonOperator(
    task_id='preprocess_pipeline_task',
    python_callable=preprocess_pipeline,
    op_args=[data_from_dvc_task.output],
    provide_context=True,
    dag=data_raw_preprocess_dag,
    on_failure_callback=email_notify_failure
)

# function to update data to dvc
update_data_to_dvc_task = PythonOperator(
    task_id = 'update_data_to_dvc_task',
    python_callable=update_data_to_dvc,
    provide_context=True,
    op_args=[preprocess_pipeline_task.output],
    dag = data_raw_preprocess_dag
)

delete_local_task = PythonOperator(
    task_id = 'delete_local_task',
    python_callable=delete_local_dvc_data,
    provide_context=True,
    dag = data_raw_preprocess_dag,
    trigger_rule=TriggerRule.ALL_DONE
)
# --------------------------

# get data from dvc (raw data) -> preprocess raw data -> push back to dvc
data_from_dvc_task >> preprocess_pipeline_task >> update_data_to_dvc_task
preprocess_pipeline_task >> [delete_local_task]
update_data_to_dvc_task >> delete_local_task 
update_data_to_dvc_task >> send_email

# ------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    data_raw_preprocess_dag.cli