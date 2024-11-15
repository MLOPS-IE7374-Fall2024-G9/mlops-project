##### Here comes the dag scripts for adding dag scripts
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
# Import required modules
import pandas as pd
import json
import pickle
import logging
# Import the provided methods
from dags.src.model_bias_detection_and_mitigation import load_splits, upload_to_gcp, run_detection, run_mitigation


# Configure logger
logger = logging.getLogger("bias_detection_mitigation")
logger.setLevel(logging.INFO)

# Define constants
BASE_PATH = "/path/to/data"
TEMP_PATH = "/path/to/temp"
GCP_BUCKET_NAME = "your-bucket-name"
MODEL_PATH = "path/to/your/model.pkl" # latest local model stored 


paths = {
            "X_train": os.path.join(BASE_PATH, "X_train.csv"),
            "X_test": os.path.join(BASE_PATH, "X_test.csv"),
            "y_train": os.path.join(BASE_PATH, "y_train.csv"),
            "y_test": os.path.join(BASE_PATH, "y_test.csv"),
            "sensitive_train": os.path.join(BASE_PATH, "sensitive_train.csv"),
            "sensitive_test": os.path.join(BASE_PATH, "sensitive_test.csv"),
        }


# Default args for DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 11, 15),
    'retries': 1,
}

# Define the DAG
with DAG(
    'bias_detection_and_mitigation',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    def load_data_splits(**kwargs):
        """
        Task to load data splits.
        """
        logger.info("Loading data splits.")
        paths = {
            "X_train": os.path.join(BASE_PATH, "X_train.csv"),
            "X_test": os.path.join(BASE_PATH, "X_test.csv"),
            "y_train": os.path.join(BASE_PATH, "y_train.csv"),
            "y_test": os.path.join(BASE_PATH, "y_test.csv"),
            "sensitive_train": os.path.join(BASE_PATH, "sensitive_train.csv"),
            "sensitive_test": os.path.join(BASE_PATH, "sensitive_test.csv"),
        }
        data = load_splits(
            paths["X_train"],
            paths["X_test"],
            paths["y_train"],
            paths["y_test"],
            paths["sensitive_train"],
            paths["sensitive_test"],
        )
        logger.info("Data splits loaded successfully.")
        kwargs['ti'].xcom_push(key='data_splits', value=data)

    def perform_bias_detection(**kwargs):
        """
        Task to perform bias detection.
        """
        logger.info("Starting bias detection.")
        ti = kwargs['ti']
        data = ti.xcom_pull(key='data_splits', task_ids='load_data_splits')
        X_test, y_test, sensitive_test = data[1], data[3], data[5]

        # Load model
        logger.info("Loading model from GCP.")
        model = pickle.load(open(MODEL_PATH, 'rb'))

        # Run detection
        detection_folder = run_detection(X_test, y_test, sensitive_test, model, GCP_BUCKET_NAME, TEMP_PATH)
        logger.info(f"Bias detection completed. Results stored in {detection_folder}.")
        ti.xcom_push(key='detection_folder', value=detection_folder)

    def perform_bias_mitigation(**kwargs):
        """
        Task to perform bias mitigation.
        """
        logger.info("Starting bias mitigation.")
        ti = kwargs['ti']
        data = ti.xcom_pull(key='data_splits', task_ids='load_data_splits')
        X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = data

        # Load model
        logger.info("Loading model from GCP.")
        model = pickle.load(open(MODEL_PATH, 'rb'))

        # Run mitigation
        mitigation_folder = run_mitigation(
            X_train, X_test, y_train, y_test, sensitive_train, sensitive_test, model, GCP_BUCKET_NAME, TEMP_PATH
        )
        logger.info(f"Bias mitigation completed. Results stored in {mitigation_folder}.")
        ti.xcom_push(key='mitigation_folder', value=mitigation_folder)

    def upload_results_to_gcp(**kwargs):
        """
        Task to upload results to GCP.
        """
        logger.info("Uploading results to GCP.")
        ti = kwargs['ti']
        detection_folder = ti.xcom_pull(key='detection_folder', task_ids='perform_bias_detection')
        mitigation_folder = ti.xcom_pull(key='mitigation_folder', task_ids='perform_bias_mitigation')

        logger.info(f"Uploading detection results from {detection_folder}.")
        upload_to_gcp(detection_folder, "detection_results", GCP_BUCKET_NAME)

        logger.info(f"Uploading mitigation results from {mitigation_folder}.")
        upload_to_gcp(mitigation_folder, "mitigation_results", GCP_BUCKET_NAME)

        logger.info("Results uploaded successfully.")

    # Define tasks
    load_data_splits_task = PythonOperator(
        task_id='load_data_splits',
        python_callable=load_data_splits,
        provide_context=True,
    )

    perform_bias_detection_task = PythonOperator(
        task_id='perform_bias_detection',
        python_callable=perform_bias_detection,
        provide_context=True,
    )

    perform_bias_mitigation_task = PythonOperator(
        task_id='perform_bias_mitigation',
        python_callable=perform_bias_mitigation,
        provide_context=True,
    )

    upload_results_to_gcp_task = PythonOperator(
        task_id='upload_results_to_gcp',
        python_callable=upload_results_to_gcp,
        provide_context=True,
    )

    # Define task dependencies
    load_data_splits_task >> perform_bias_detection_task >> perform_bias_mitigation_task >> upload_results_to_gcp_task
