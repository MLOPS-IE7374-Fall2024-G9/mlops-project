from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
from src.model_pipeline import (
    load_processed_data,
    download_model_artifacts,
    load_model,
    test_and_evaluate_model,
    threshold_verification,
)

# ------------------------------------------------------------------------------------------------
# Default Arguments
default_args = {
    'owner': 'Group 9',
    'start_date': datetime(2023, 9, 17),
    'retries': 0,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
    "execution_timeout": timedelta(minutes=10),
}

# DAG Definition
with DAG(
    "deployment_monitoring_dag",
    default_args=default_args,
    description="Dag for monitoring deployed models and triggering retraining if thresholds are violated.",
    schedule_interval='@weekly',
    catchup=False,
    tags=['conditional_retrain_dag'],
) as dag:

    # Task: Download Model
    download_model_task = PythonOperator(
        task_id="download_model_task",
        python_callable=download_model_artifacts,
    )

    # Task: Trigger Data Drift Detection DAG
    trigger_data_drift_dag = TriggerDagRunOperator(
        task_id="trigger_data_drift_dag",
        trigger_dag_id="drift_data_dag",
        wait_for_completion=True,
    )

    # Task: Trigger Data Bias Mitigation DAG
    trigger_data_bias_dag = TriggerDagRunOperator(
        task_id="trigger_data_bias_dag",
        trigger_dag_id="bias_detection_and_mitigation",
        wait_for_completion=True,
    )

    # Task: Trigger Model Bias Detection DAG
    trigger_model_bias_dag = TriggerDagRunOperator(
        task_id="trigger_model_bias_dag",
        trigger_dag_id="model_bias_detection_and_mitigation",
        wait_for_completion=True,
    )

    # Task: Get Validation Outputs
    def get_validation_outputs(**kwargs):
        X_train, X_test, y_train, y_test = load_processed_data(
            filename="data_preprocess.csv",
            target_column="value",
            test_size=0.2,
            random_state=42,
        )
        if X_test is None or y_test is None:
            raise ValueError("Failed to load the test dataset.")
        
        model_name = download_model_artifacts()
        model = load_model(model_name)
        
        metrics = test_and_evaluate_model(model, X_test, y_test)
        return metrics["mse"], metrics["mae"], metrics["r2 score"]

    get_validation_outputs_task = PythonOperator(
        task_id="get_validation_outputs_task",
        python_callable=get_validation_outputs,
    )

    # Task: Check Thresholds - skip or trigger rollback + retraining
    def check_thresholds_task(**kwargs):
        thresholds = kwargs['thresholds']
        validation_outputs = kwargs['ti'].xcom_pull(task_ids="get_validation_outputs_task")
        result = threshold_verification(thresholds, validation_outputs)
        return "skip_retraining" if result else "trigger_rollback_dag"

    check_thresholds = BranchPythonOperator(
        task_id="check_thresholds",
        python_callable=check_thresholds_task,
        op_kwargs={
            'thresholds': (1000, 1000, 0.7),
        },
    )

    # Task: Skip Retraining
    def skip_retraining_task():
        print("Skipping retraining. Thresholds met.")

    skip_retraining = PythonOperator(
        task_id="skip_retraining",
        python_callable=skip_retraining_task,
    )

    # Task: Trigger Retraining DAG
    trigger_retraining_dag = TriggerDagRunOperator(
        task_id="trigger_retraining_dag",
        trigger_dag_id="model_retrain_evaluate",
        wait_for_completion=True,
    )

    # Task: Trigger rollback
    trigger_rollback_dag = TriggerDagRunOperator(
        task_id="trigger_rollback_dag",
        trigger_dag_id="model_rollback",
        wait_for_completion=True,
    )

    # DAG Dependencies
    download_model_task >> trigger_data_drift_dag >> trigger_data_bias_dag >> trigger_model_bias_dag
    trigger_model_bias_dag >> get_validation_outputs_task >> check_thresholds
    check_thresholds >> [skip_retraining, trigger_rollback_dag]
    trigger_rollback_dag >> trigger_retraining_dag
