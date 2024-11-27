from airflow import DAG
from airflow.operators.email_operator import EmailOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

deploy_model_path = os.path.join(os.path.dirname(__file__), '../setup-scripts/deploy_model.sh')
deploy_app_path = os.path.join(os.path.dirname(__file__), '../setup-scripts/deploy_app.sh')

# default args
default_args = {
    'owner': 'Group 9',
    'start_date': datetime(2023, 9, 17),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
    "execution_timeout": timedelta(minutes=10),
}

# Data DAG pipeline init
deploy_model_dag = DAG(
    "deploy_model_dag",
    default_args=default_args,
    description="Training Model on incoming data",
    schedule_interval=None,
    catchup=False,
    tags=['deploy_model_dag']
)

# -------------------------------------------------
# Task to send success email
deployment_pass_email = EmailOperator(
    task_id='send_deployment_pass_email',
    to=["mlops.group.9@gmail.com"],
    subject='Deployment Task Success',
    html_content='<p>Deployment completed successfully.</p>',
    trigger_rule=TriggerRule.ALL_SUCCESS,  # Trigger only if the previous tasks succeeded
    dag=deploy_model_dag,
)

# Task to send failure email
deployment_fail_email = EmailOperator(
    task_id='send_deployment_failure_email',
    to=["mlops.group.9@gmail.com"],
    subject='Deployment Task Failure',
    html_content='<p>Deployment failed. Please check the logs for details.</p>',
    trigger_rule=TriggerRule.ONE_FAILED,  # Trigger if any of the previous tasks failed
    dag=deploy_model_dag,
)
# -------------------------------------------------
# Task to execute `deploy_model.sh`
deploy_model_task = BashOperator(
    task_id='deploy_model',
    bash_command='bash ' + deploy_model_path,
    dag=deploy_model_dag,
)

# Task to execute `deploy_app.sh`
deploy_app_task = BashOperator(
    task_id='deploy_app',
    bash_command='bash ' + deploy_app_path,
    dag=deploy_model_dag,
)

# Task dependencies
deploy_model_task >> deploy_app_task >> [deployment_pass_email, deployment_fail_email]

