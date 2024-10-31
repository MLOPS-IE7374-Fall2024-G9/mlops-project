import os
import sys

# # Add the path to the 'dataset' directory
# sys.path.insert(0, os.path.abspath('/opt/airflow/dataset'))

# # Add the path to the 'src' directory inside 'dags'
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


from airflow import DAG
from airflow.operators.python_operator import PythonOperator
# from airflow.operators.email_operator import EmailOperator
from datetime import datetime, timedelta
from src.data_preprocess import *
from src.data_pipeline import *

# default args
default_args = {
    'owner': 'Group 9',
    'start_date': datetime(2023, 9, 17),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
}

# Data Preprocess DAG pipeline
dag = DAG(
    "data_preprocess_dag",
    default_args=default_args,
    description="A data preprocessing pipeline DAG",
    schedule_interval=None,  # Do not schedule automatically, can be triggered manually
    catchup=False,
    tags=['data_preprocess_dag']
)

# Define PythonOperators for each function

#Define the dvc data pull task
get_data_from_dvc_task = PythonOperator(
    task_id = 'get_data_from_dvc_task',
    python_callable=get_data_from_dvc,
    provide_context=True,
    dag = dag
)

# Define the clean data task, depends on 'get_data_from_dvc_task'
clean_data_task = PythonOperator(
    task_id='clean_data_task',
    python_callable=clean_data,
    op_args=[get_data_from_dvc_task.output],
    provide_context=True,
    dag=dag,
)

save_clean_data_task = PythonOperator(
    task_id='save_clean_data_task',
    python_callable=save_data,
    provide_context=True,
    op_args=[clean_data_task.output,"cleaned_data"],
    dag=dag,
)

# Define the engineer features task, depends on 'clean_data_task'
engineer_features_task = PythonOperator(
    task_id='engineer_features_task',
    python_callable=engineer_features,
    op_args=[clean_data_task.output],
    provide_context=True,
    dag=dag,
)

# Define the cyclic feature addition task, depends on 'engineer_features_task'
add_cyclic_features_task = PythonOperator(
    task_id='add_cyclic_features_task',
    python_callable=add_cyclic_features,
    op_args=[engineer_features_task.output],
    provide_context=True,
    dag=dag,
)

# Define the normalization and encoding task, depends on 'add_cyclic_features_task'
normalize_and_encode_task = PythonOperator(
    task_id='normalize_and_encode_task',
    python_callable=normalize_and_encode,
    op_args=[add_cyclic_features_task.output],
    provide_context=True,
    dag=dag,
)

# Define the feature selection task, depends on 'normalize_and_encode_task'
select_final_features_task = PythonOperator(
    task_id='select_final_features_task',
    python_callable=select_final_features,
    op_args=[normalize_and_encode_task.output],
    provide_context=True,
    dag=dag,
)

save_selected_data_task = PythonOperator(
    task_id='save_selected_data_task',
    python_callable=save_data,
    provide_context=True,
    op_args=[select_final_features_task.output,"final_selected_features"],
    dag=dag,
)

# Define task dependencies
get_data_from_dvc_task >> clean_data_task >> save_clean_data_task >> engineer_features_task >> add_cyclic_features_task >> normalize_and_encode_task >> select_final_features_task >> save_selected_data_task