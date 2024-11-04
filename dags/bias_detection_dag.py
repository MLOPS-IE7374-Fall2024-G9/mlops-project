from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from src.data_bias_detection import detect_bias
from src.data_bias_detection import conditional_mitigation
import pandas as pd
import pickle

# Import necessary functions or define them here if not imported
# from my_bias_module import detect_bias, conditional_mitigation
from src.data_download import *

default_args = {
    'owner': 'user',
    'start_date': datetime(2024, 11, 1),
    'retries': 1,
}

bias_detection_and_mitigation = DAG(
    'bias_detection_and_mitigation',
    default_args=default_args,
    description='A DAG with separate tasks for bias detection and mitigation',
    schedule_interval='@daily',
)

# File paths for data
data_path = '/opt/airflow/dataset/data/data_preprocess.csv'
bias_results_path = '/opt/airflow/model/bias_detection_results.pkl'  # For storing intermediate results
mitigated_data_path = '/opt/airflow/dataset/data/bias_mitigated_data.csv'

filename_preprocessed = "data_preprocess.csv"


def identify_bias(data_path):
    # Load the new data
    data = pd.read_csv(data_path)
    target_col = 'value'
    sensitive_col = 'subba-name'

    # Detect bias in the data
    bias_output = detect_bias(data, target_col, sensitive_col)

    # Save the results to a file for later use in the second task
    with open(bias_results_path, 'wb') as f:
        pickle.dump(bias_output, f)

def mitigate_bias():
    # Load the new data and bias results
    data = pd.read_csv(data_path)

    with open(bias_results_path, 'rb') as f:
        bias_output = pickle.load(f)

    print(bias_output)
    
    target_col = 'value'
    sensitive_col = 'subba-name'

    # Perform conditional mitigation based on the results of the bias detection
    mitigated_data = conditional_mitigation(data, target_col, sensitive_col, bias_output)
    
    # Save the mitigated data
    mitigated_data.to_csv(mitigated_data_path, index=False)

# Define the tasks
processed_data_from_dvc_task = PythonOperator(
    task_id = 'processed_data_from_dvc_task',
    python_callable=get_data_from_dvc,
    provide_context=True,
    op_args=[filename_preprocessed],
    dag = bias_detection_and_mitigation
)

identify_bias_task = PythonOperator(
    task_id='identify_bias',
    python_callable=identify_bias,
    op_args=[processed_data_from_dvc_task.output],
    dag=bias_detection_and_mitigation,
)

mitigate_bias_task = PythonOperator(
    task_id='mitigate_bias',
    python_callable=mitigate_bias,
    dag=bias_detection_and_mitigation,
)

# function to update data to dvc
mitigated_data_to_dvc_task = PythonOperator(
    task_id = 'mitigated_data_to_dvc_task',
    python_callable=update_data_to_dvc,
    provide_context=True,
    op_args=[mitigated_data_path],
    dag = bias_detection_and_mitigation
)

# Set task dependencies
processed_data_from_dvc_task >> identify_bias_task >> mitigate_bias_task >> mitigated_data_to_dvc_task
 
# ------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    bias_detection_and_mitigation.cli
