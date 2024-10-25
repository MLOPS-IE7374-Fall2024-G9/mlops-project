import os
import sys
# testing git workflow for dag folder update 3
# Add the path to the 'dataset' directory
sys.path.insert(0, os.path.abspath('/opt/airflow/dataset'))

# Add the path to the 'src' directory inside 'dags'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime, timedelta
from src.data_pipeline import *

# default args
default_args = {
    'owner': 'Group 9',
    'start_date': datetime(2023, 9, 17),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
}

# Data DAG pipeline init
data_dag = DAG(
    "Data_Pipeline_Dag",
    default_args=default_args,
    description="Data Pipeline DAG init",
    schedule_interval=None,  # Do not schedule automatically, can be triggered manually
    catchup=False,
    tags=['data_pipeline']
)


# ------------------------------------------------------------------------------------------------
# Email operators
def email_notify_success(context):
    success_email = EmailOperator(
        task_id='success_email',
        to='keshri.r@northeastern.edu',
        subject='Task Success',
        html_content='<p>The task succeeded.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

def email_notify_failure(context):
    success_email = EmailOperator(
        task_id='failure_email',
        to='keshri.r@northeastern.edu',
        subject='Task Failure',
        html_content='<p>The task succeeded.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

# send_email = EmailOperator(
#     task_id='send_email',
#     to='keshri.r@northeastern.edu',    # Email address of the recipient
#     subject='Notification from Airflow',
#     html_content='<p>This is a notification email sent from Airflow.</p>',
#     dag=data_dag,
#     on_failure_callback=email_notify_failure,
#     on_success_callback=email_notify_success
# )

# ------------------------------------------------------------------------------------------------
# Python operators
# --------------------------
# Data API Operators
# function to pull data from dvc, returns json
get_data_from_dvc_task = PythonOperator(
    task_id = 'get_data_from_dvc_task',
    python_callable=get_data_from_dvc,
    provide_context=True,
    dag = data_dag
)

# function returns start and end date (yesterday's date)
get_start_end_date_task = PythonOperator(
    task_id = 'get_start_end_date_task',
    python_callable=get_start_end_dates,
    provide_context=True,
    dag = data_dag
)

# function to get new data, returns the json
get_updated_data_from_api_task = PythonOperator(
    task_id = 'get_updated_data_from_api_task',
    python_callable=get_updated_data_from_api,
    provide_context=True,
    op_args=[get_start_end_date_task.output],
    dag = data_dag
)

# function to merge dvc data with newly pulled data from api
merge_data_task = PythonOperator(
    task_id = 'merge_data_task',
    python_callable=merge_data,
    provide_context=True,
    op_args=[get_updated_data_from_api_task.output, get_data_from_dvc_task.output],
    dag = data_dag
)

# function to update data to dvc
update_data_to_dvc_task = PythonOperator(
    task_id = 'update_data_to_dvc_task',
    python_callable=update_data_to_dvc,
    provide_context=True,
    op_args=[merge_data_task.output],
    dag = data_dag
)
# --------------------------




# ------------------------------------------------------------------------------------------------
# Data DAG Pipelines

# 1) data api pipeline
get_data_from_dvc_task >> get_start_end_date_task >> get_updated_data_from_api_task >> merge_data_task >> update_data_to_dvc_task

# 2) data preprocessing pipeline

# 3) data drift pipeline

# ------------------------------------------------------------------------------------------------
# Model DAG Pipelines TODO


# ------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    data_dag.cli