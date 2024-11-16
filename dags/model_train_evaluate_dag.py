
from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta

from src.data_download import *
from src.model_pipeline import *

#### dag 1
# pull from dvc
# load model 
## if no model - train from scratch - entire dataset
## train model
## evaluate model
## save the model

## if model - fine tune, get date until which it was trained
## split dataset
## train model only for dataset split based on dates
## evaluate model
## save the model

# manual trigger
# check evaluation metrics
## if less than threshold -> hyperparamater fine tuning
## retraining
## evaluatiion
## save
####

# variables 
filename = "data_preprocess.csv"
model_name = "xgboost"
thresholds = (1000, 1000, 0.7)

# default args
default_args = {
    'owner': 'Group 9',
    'start_date': datetime(2023, 9, 17),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
    "execution_timeout": timedelta(minutes=10),
}

# Data DAG pipeline init
model_train_evaluate = DAG(
    "model_train_evaluate",
    default_args=default_args,
    description="Training Model on incoming data",
    schedule_interval=None,
    catchup=False,
    tags=['model_train_evaluate']
)


# --------------------------
# email for training done
# train_pass_email = EmailOperator(
#     task_id='send_train_pass_email',
#     to=["mlops.group.9@gmail.com"],
#     subject='Training done',
#     html_content='<p>Training done.</p>',
#     dag=model_train_evaluate,
# )

# train_fail_email = EmailOperator(
#     task_id='send_train_failure_email',
#     to=["mlops.group.9@gmail.com"],
#     subject='Training failed',
#     html_content='<p>Training failed.</p>',
#     dag=model_train_evaluate,
# )

threshold_pass_email = EmailOperator(
    task_id='send_threshold_pass_email',
    to=["mlops.group.9@gmail.com"],
    subject='threshold task success',
    html_content='<p>thresholds met.</p>',
    trigger_rule=TriggerRule.ALL_SUCCESS,  # Trigger only if the previous task succeeded
    dag=model_train_evaluate,
)

# Task to send email if training failed
threshold_fail_email = EmailOperator(
    task_id='send_threshold_failure_email',
    to=["mlops.group.9@gmail.com"],
    subject='threshold task failed',
    html_content='<p>thresholds not met.</p>',
    trigger_rule=TriggerRule.ALL_FAILED,  # Trigger only if the previous task failed
    dag=model_train_evaluate,
)

# --------------------------
def choose_task_based_on_trigger(**kwargs):
    train_from_scratch = True #kwargs['dag_run'].conf.get('train_from_scratch', 'false') == 'true'
    if train_from_scratch:
        return 'train_on_all_data_task'  # Train on all data
    else:
        return 'download_model_task'  # fine tune

# --------------------------
# pull from dvc - returns the filepath where the data is
data_from_dvc_task = PythonOperator(
    task_id = 'download_data_from_dvc_task',
    python_callable=get_data_from_dvc,
    provide_context=True,
    op_args=[filename],
    dag = model_train_evaluate
)


# choosing to train from scratch or finetuning
choose_task = BranchPythonOperator(
    task_id='choose_task',
    python_callable=choose_task_based_on_trigger,
    provide_context=True,
    dag=model_train_evaluate
)


# load model - downloads all the models and returns the model path
download_model_task = PythonOperator(
    task_id = 'download_model_task',
    python_callable=download_model_artifacts,
    provide_context=True,
    dag = model_train_evaluate
)

# train the model -> save in local
train_on_all_data_task = PythonOperator(
    task_id = 'train_on_all_data_task',
    python_callable=train_model,
    provide_context=True,
    op_args=[data_from_dvc_task.output, model_name],
    dag = model_train_evaluate
)

# fine tune the model on new data -> takes input data file path and model file path -> finetunes and saves in local -> return model path
fine_tune_on_new_data_task = PythonOperator(
    task_id = 'fine_tune_on_new_data_task',
    python_callable=train_model,
    provide_context=True,
    op_args=[data_from_dvc_task.output, download_model_task.output, True],
    dag = model_train_evaluate
)

# evaluate the model -> returns the evaluation metrics and model path
evaluate_model_task = PythonOperator(
    task_id = 'evaluate_model_task',
    python_callable=validate_model,
    provide_context=True,
    op_args=[fine_tune_on_new_data_task.output, train_on_all_data_task.output, data_from_dvc_task.output],
    dag = model_train_evaluate, 
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
)

# check evaluation with respect to thresholds
threshold_check_task = PythonOperator(
    task_id = 'threshold_check_task',
    python_callable=threshold_verification,
    provide_context=True,
    op_args=[thresholds, evaluate_model_task.output],
    dag = model_train_evaluate,
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
)

delete_local_task = PythonOperator(
    task_id = 'delete_local_task',
    python_callable=delete_local_model_data,
    provide_context=True,
    dag = model_train_evaluate,
    trigger_rule=TriggerRule.ALL_DONE
)

data_from_dvc_task >> choose_task >> [train_on_all_data_task, download_model_task]
train_on_all_data_task >> evaluate_model_task >> threshold_check_task >> [threshold_pass_email, threshold_fail_email] >> delete_local_task
download_model_task >> fine_tune_on_new_data_task >> evaluate_model_task >> threshold_check_task >> [threshold_pass_email, threshold_fail_email] >> delete_local_task

if __name__ == "__main__":
    model_train_evaluate.cli