from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
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
new_data_dag = DAG(
    "new_data_dag",
    default_args=default_args,
    description="New Data Download and Preprocess DAG",
    schedule_interval=None,
    catchup=False,
    tags=['new_data_dag']
)

# ------------------------------------------------------------------------------------------------
# variables
delta_days = 7
filename_raw = "data_raw.csv"
filename_preprocessed = "data_preprocess.csv"

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
# function returns start and end date for last "number of days"
last_k_start_end_date_task = PythonOperator(
    task_id = 'last_k_start_end_date_task',
    python_callable=get_last_k_start_end_dates,
    provide_context=True,
    op_args=[delta_days],
    dag = new_data_dag
)

# function returns start and end date (yesterday's date)
# start_end_date_task = PythonOperator(
#     task_id = 'start_end_date_task',
#     python_callable=get_start_end_dates,
#     provide_context=True,
#     dag = new_data_dag
# )

# function to get new data, returns data json
updated_data_from_api_task = PythonOperator(
    task_id = 'updated_data_from_api_task',
    python_callable=get_updated_data_from_api,
    provide_context=True,
    op_args=[last_k_start_end_date_task.output],
    dag = new_data_dag
)

# Define the clean data task, depends on 'updated_data_from_api_task'
clean_data_task = PythonOperator(
    task_id='clean_data_task',
    python_callable=clean_data,
    op_args=[updated_data_from_api_task.output],
    provide_context=True,
    dag=new_data_dag,
)

# Define the engineer features task, depends on 'clean_data_task'
engineer_features_task = PythonOperator(
    task_id='engineer_features_task',
    python_callable=engineer_features,
    op_args=[clean_data_task.output],
    provide_context=True,
    dag=new_data_dag,
)

# Define the cyclic feature addition task, depends on 'engineer_features_task'
add_cyclic_features_task = PythonOperator(
    task_id='add_cyclic_features_task',
    python_callable=add_cyclic_features,
    op_args=[engineer_features_task.output],
    provide_context=True,
    dag=new_data_dag,
)

# Define the normalization and encoding task, depends on 'add_cyclic_features_task'
normalize_and_encode_task = PythonOperator(
    task_id='normalize_and_encode_task',
    python_callable=normalize_and_encode,
    op_args=[add_cyclic_features_task.output],
    provide_context=True,
    dag=new_data_dag,
)

select_final_features_task = PythonOperator(
    task_id='select_final_features_task',
    python_callable=select_final_features,
    op_args=[normalize_and_encode_task.output],
    provide_context=True,
    dag=new_data_dag,
)

# function to pull preprocessed data from dvc, returns json
data_from_dvc_task = PythonOperator(
    task_id = 'data_from_dvc_task',
    python_callable=get_data_from_dvc,
    provide_context=True,
    op_args=[filename_preprocessed],
    dag = new_data_dag
)

# function to merge dvc data with newly pulled data from api, returns data json
merge_data_task = PythonOperator(
    task_id = 'merge_data_task',
    python_callable=merge_data,
    provide_context=True,
    op_args=[select_final_features_task.output, data_from_dvc_task.output],
    dag = new_data_dag
)

# function to remove redundant rows, returns data json
redundant_removal_task = PythonOperator(
    task_id = 'redundant_removal_task',
    python_callable=redundant_removal,
    provide_context=True,
    op_args=[merge_data_task.output],
    dag = new_data_dag
)

# function to update data to dvc
update_data_to_dvc_task = PythonOperator(
    task_id = 'update_data_to_dvc_task',
    python_callable=update_data_to_dvc,
    provide_context=True,
    op_args=[redundant_removal_task.output],
    dag = new_data_dag
)

delete_local_task = PythonOperator(
    task_id = 'delete_local_task',
    python_callable=delete_local_dvc_data,
    provide_context=True,
    dag = new_data_dag
)
# --------------------------

# get data from api (new data) -> get data from dvc -> merge new data with dvc -> push back to dvc
data_from_dvc_task >> last_k_start_end_date_task >> updated_data_from_api_task >> clean_data_task >> engineer_features_task >> add_cyclic_features_task >> normalize_and_encode_task >> select_final_features_task >> merge_data_task >> redundant_removal_task >> update_data_to_dvc_task #>> delete_local_task

# ------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    new_data_dag.cli