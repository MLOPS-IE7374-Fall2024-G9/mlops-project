from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta


# pull latest deployed model
# data drift, model decay detection
# threshold detection
# retraining trigger 
