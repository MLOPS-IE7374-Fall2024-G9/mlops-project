
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
mse_threshold = 1000  # Example MSE threshold for model evaluation
r2_threshold = 0.7  # Example R2 threshold for model evaluation
model_name = "xgboost"

# default args
default_args = {
    'owner': 'Group 9',
    'start_date': datetime.datetime(2023, 9, 17),
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
def evaluate_model_and_branch(**kwargs):
    model_path = kwargs['ti'].xcom_pull(task_ids='download_model_task')
    data_path = kwargs['ti'].xcom_pull(task_ids='data_from_dvc_task')
        
    # TODO
    # load the model
    # load the data - get the xtest and ytest

    eval_metrics = test_and_evaluate_model(model, X_test, y_test)  # Custom evaluation function
    mse = eval_metrics["mse"]
    r2 = eval_metrics["r2 score"]
    
    # Save the evaluation metrics in XCom for further use
    kwargs['ti'].xcom_push(key="mse", value=mse)
    kwargs['ti'].xcom_push(key="r2", value=r2)
    
    # Branch based on the thresholds
    if mse > mse_threshold or r2 < r2_threshold:
        return 'no_save_model_task'  # Branch to no save model task
    else:
        return 'save_model_task'  # Continue to save model task

def choose_task_based_on_trigger(**kwargs):
    train_from_scratch = False #kwargs['dag_run'].conf.get('train_from_scratch', 'false') == 'true'
    if train_from_scratch:
        return 'train_on_all_data_task'  # Train on all data
    else:
        return 'download_model_task'  # Download the model
    
def branch_model_evaluation(**kwargs):
    # Check if the model was trained or fine-tuned
    model_path = kwargs['ti'].xcom_pull(task_ids='train_on_all_data_task')  # Pull from train_on_all_data_task
    if model_path:
        # If the model was trained from scratch, return the model path
        return 'evaluate_model_task'
    else:
        # If the model was fine-tuned, return the fine-tuned model path
        return 'fine_tune_and_evaluate_task'

# --------------------------
# pull from dvc - returns the filepath where the data is
data_from_dvc_task = PythonOperator(
    task_id = 'data_from_dvc_task',
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

branch_model_evaluation_task = BranchPythonOperator(
    task_id='branch_model_evaluation_task',
    python_callable=branch_model_evaluation,
    provide_context=True,
    dag=model_train_evaluate
)



# load model - downloads the model and returns the model path
download_model_task = PythonOperator(
    task_id = 'download_model_task',
    python_callable=download_model_from_gcs,
    provide_context=True,
    op_args=[model_name],
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
    task_id = 'train_on_all_data_task',
    python_callable=train_model,
    provide_context=True,
    op_args=[data_from_dvc_task.output, download_model_task.output],
    dag = model_train_evaluate
)

# evaluate the model -> returns the evaluation metrics and model path
evaluate_model_task = PythonOperator(
    task_id = 'evaluate_model_task',
    python_callable=validate_model,
    provide_context=True,
    op_args=[fine_tune_on_new_data_task.output, train_on_all_data_task.output, data_from_dvc_task.output],
    dag = model_train_evaluate
)


data_from_dvc_task >> choose_task  # Branching decision
choose_task >> [train_on_all_data_task, download_model_task] >> branch_model_evaluation_task# Based on flag, either train or download

branch_model_evaluation_task >> evaluate_model_task  # If training from scratch, evaluate
branch_model_evaluation_task >> fine_tune_on_new_data_task >> evaluate_model_task  # If loading model, fine tune and evaluate


if __name__ == "__main__":
    model_train_evaluate.cli