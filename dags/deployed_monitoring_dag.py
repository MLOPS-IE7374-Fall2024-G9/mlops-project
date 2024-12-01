from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
from src.model_pipeline import *


# pull latest deployed model
# data drift, model decay detection
# threshold detection
# retraining trigger 




# ------------------------------------------------------------------------------------------------

# variables 
filename = "data_preprocess.csv"
model_name = "xgboost"


# Function to get validation outputs
def get_validation_outputs(**kwargs):
    # Load processed data
    X_train, X_test, y_train, y_test = load_processed_data(
        filename="data_preprocess.csv",
        target_column="value",
        test_size=0.2,
        random_state=42
    )
    if X_test is None or y_test is None:
        raise ValueError("Failed to load the test dataset.")
    
    # Download and load the model
    model_name = download_model_artifacts()
    model = load_model(model_name)
    
    # Evaluate the model
    metrics = test_and_evaluate_model(model, X_test, y_test)
    
    # Extract metrics
    mse = metrics["mse"]
    r2 = metrics["r2 score"]
    mae = metrics["mae"] 
    return mse, mae, r2



# Function to execute the threshold verification task and decide branching
def check_thresholds_task(**kwargs):
    thresholds = kwargs['thresholds']
    
    # Retrieve validation outputs from XCom (previous task)
    validation_outputs = kwargs['ti'].xcom_pull(task_ids="get_validation_outputs")
    
    # Call the threshold_verification function
    result = threshold_verification(thresholds, validation_outputs)

    # Log results and return task ID
    if result:
        print("Model is within thresholds. No retraining needed.")
        return "skip_retraining"
    else:
        print("Model is outside thresholds. Retraining may be required.")
        return "trigger_retraining_dag"


# Dummy task to handle the "no retraining" scenario
def skip_retraining_task():
    print("Skipping retraining. Thresholds met.")



# default args
default_args = {
    'owner': 'Group 9',
    'start_date': datetime.datetime(2023, 9, 17),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
    "execution_timeout": timedelta(minutes=10),
}

# Data DAG pipeline init
deployment_monitoring_dag = DAG(
    "deployment_monitoring_dag",
    default_args=default_args,
    description="Dag for monitoring of the different models and triggers retraining based on thresholds",
    schedule_interval='@daily',
    catchup=False,
    tags=['conditional_retrain_dag']
)
# ------------------------------------------------------------------------------------------------

# load model - downloads all the models and returns the model path
download_model_task = PythonOperator(
    task_id = 'download_model_task',
    python_callable=download_model_artifacts,
    provide_context=True
)

# data drift check
trigger_data_drift_dag = TriggerDagRunOperator(
    task_id="trigger_data_drift_dag",
    trigger_dag_id="drift_data_dag",  
    wait_for_completion=True,
)
# data bias mitigation
trigger_data_bias_dag = TriggerDagRunOperator(
    task_id="trigger_data_bias_dag",
    trigger_dag_id="bias_detection_and_mitigation",  
    wait_for_completion=True,
)
# model bias detection and mitigation
trigger_model_bias_dag = TriggerDagRunOperator(
    task_id="trigger_model_bias_dag",
    trigger_dag_id="model_bias_detection_and_mitigation",  
    wait_for_completion=True,
)

# get validation outputs
get_validation_outputs_task = PythonOperator(
    task_id="get_validation_outputs_task",
    python_callable=get_validation_outputs,
)

# check thresholds
check_thresholds = PythonOperator(
    task_id="check_thresholds",
    python_callable=check_thresholds_task,
    op_kwargs={
        'thresholds': (1000, 1000, 0.7),
    },
)

# Skip retraining
skip_retraining = PythonOperator(
    task_id="skip_retraining",
    python_callable=skip_retraining_task,
)

# Trigger retraining DAG
trigger_retraining_dag = TriggerDagRunOperator(
    task_id="trigger_retraining_dag",
    trigger_dag_id="model_retrain_evaluate",  
    wait_for_completion=True,
)

download_model_task >> trigger_data_drift_dag >> trigger_data_bias_dag >> trigger_model_bias_dag >> get_validation_outputs_task >> check_thresholds >> [skip_retraining, trigger_retraining_dag]
