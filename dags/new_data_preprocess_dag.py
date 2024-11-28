from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
from src.data_download import *
from src.data_preprocess import *
from src.data_schema_validation import *


# ------------------------------------------------------------------------------------------------
# variables
delta_days = 7
filename_preprocessed = "data_preprocess.csv"
filename_raw = "data_raw.csv"

# local functions
# Function to determine whether to proceed based on validation result
def check_validation_result(**kwargs):
    validation_result = kwargs['ti'].xcom_pull(task_ids='validate_data_with_schema_task')
    if validation_result == 1:
        return 'merge_data_task'  # Task to continue if validation is successful
    else:
        return 'send_failure_email'  # Task to send failure email and stop flow


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
    description="New Data Download and Preprocess DAG",
    schedule_interval='@daily',
    catchup=False,
    tags=['new_data_dag']
)

# ------------------------------------------------------------------------------------------------
# Email operator
def email_notify_success(context):
    success_email = EmailOperator(
        task_id='success_email',
        to=["mlops.group.9@gmail.com"],
        subject='Task Success',
        html_content='<p>New data download succeeded.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

def email_notify_failure(context):
    success_email = EmailOperator(
        task_id='failure_email',
        to=["mlops.group.9@gmail.com"],
        subject='Task Failure',
        html_content='<p>New data download task Failed.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

send_email = EmailOperator(
    task_id='send_email',
    to=["mlops.group.9@gmail.com"],    # Email address of the recipient
    subject='Notification from Airflow',
    html_content='<p>This is a notification email sent from Airflow. </p>',
    dag=new_data_dag,
    on_failure_callback=email_notify_failure,
    on_success_callback=email_notify_success, 
)

# Task to send notification email for validation failure
send_data_validation_failure_email = EmailOperator(
    task_id='send_failure_email',
    to=["mlops.group.9@gmail.com"],
    subject='Data Validation Failed',
    html_content='<p>Data validation has failed. Please review the data.</p>',
    dag=new_data_dag,
)

# ------------------------------------------------------------------------------------------------
# Python operators
# Branch operator to decide the next step based on validation result
branch_task = BranchPythonOperator(
    task_id='branch_on_validation',
    python_callable=check_validation_result,
    provide_context=True,
    dag=new_data_dag,
)

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

# function to select final features
select_final_features_task = PythonOperator(
    task_id='select_final_features_task',
    python_callable=select_final_features,
    op_args=[normalize_and_encode_task.output],
    provide_context=True,
    dag=new_data_dag,
)


# function to pull preprocessed data from dvc, returns json
processed_data_from_dvc_task = PythonOperator(
    task_id = 'processed_data_from_dvc_task',
    python_callable=get_data_from_dvc,
    provide_context=True,
    op_args=[filename_preprocessed],
    dag = new_data_dag
)

# function to validate data with schema
validate_data_with_schema_task =  PythonOperator(
    task_id = 'validate_data_with_schema_task',
    python_callable=validate_data,
    provide_context=True,
    op_args=[processed_data_from_dvc_task.output, select_final_features_task.output],
    dag = new_data_dag
)

# function to pull preprocessed data from dvc, returns json
raw_data_from_dvc_task = PythonOperator(
    task_id = 'raw_data_from_dvc_task',
    python_callable=get_data_from_dvc,
    provide_context=True,
    op_args=[filename_raw],
    dag = new_data_dag
)

# function to merge the new data to raw data csv directly
merge_raw_data_task = PythonOperator(
    task_id = 'merge_raw_data_task',
    python_callable=merge_data,
    provide_context=True,
    op_args=[updated_data_from_api_task.output, raw_data_from_dvc_task.output],
    dag = new_data_dag
)

# function to merge dvc data with newly pulled data from api, returns data json
merge_data_task = PythonOperator(
    task_id = 'merge_data_task',
    python_callable=merge_data,
    provide_context=True,
    op_args=[select_final_features_task.output, processed_data_from_dvc_task.output],
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

update_raw_data_to_dvc_task = PythonOperator(
    task_id = 'update_raw_data_to_dvc_task',
    python_callable=update_data_to_dvc,
    provide_context=True,
    op_args=[merge_raw_data_task.output],
    dag = new_data_dag
)

delete_local_task = PythonOperator(
    task_id = 'delete_local_task',
    python_callable=delete_local_dvc_data,
    provide_context=True,
    dag = new_data_dag,
    trigger_rule=TriggerRule.ALL_DONE
)

# Trigger the bias detection DAG at the end
trigger_bias_detection_dag = TriggerDagRunOperator(
    task_id="trigger_bias_detection",
    trigger_dag_id="bias_detection_and_mitigation",  # Name of the second DAG to trigger
    wait_for_completion=False,    # Set to True if you want to wait for the second DAG to complete
)

# --------------------------

# get data from api (new data) -> get data from dvc -> merge new data with dvc -> push back to dvc
last_k_start_end_date_task >> updated_data_from_api_task >> clean_data_task >> engineer_features_task >> add_cyclic_features_task >> normalize_and_encode_task >> select_final_features_task >> processed_data_from_dvc_task >> validate_data_with_schema_task >> branch_task
branch_task >> merge_data_task >> redundant_removal_task >> update_data_to_dvc_task
branch_task >> send_data_validation_failure_email
raw_data_from_dvc_task >> merge_raw_data_task >> update_raw_data_to_dvc_task
[update_data_to_dvc_task , update_raw_data_to_dvc_task] >> delete_local_task #>> trigger_bias_detection_dag
update_raw_data_to_dvc_task >> send_email 



# ------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    new_data_dag.cli
