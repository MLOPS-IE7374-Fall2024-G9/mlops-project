import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dataset.scripts.data_drift_detection import *
from dataset.scripts.dvc_manager import *
from dags.src.data_download import get_data_from_dvc

# Task to load data
def load_data(filename, **kwargs):
    file_path = get_data_from_dvc(filename)
    df = pd.read_csv(file_path)

    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Sort the data by datetime (if not already sorted)
    df = df.sort_values(by='datetime').reset_index(drop=True)

    # Get the last available date
    last_available_date = df['datetime'].max()

    # Split the data
    baseline_df = df[df['datetime'] < last_available_date]  # All data except the last available date
    new_data_df = df[df['datetime'] == last_available_date]

    kwargs['ti'].xcom_push(key='baseline_df', value=baseline_df.to_dict())
    kwargs['ti'].xcom_push(key='new_data_df', value=new_data_df.to_dict())

# Initialize DataDriftDetector and call Evidently drift detection
def detect_drift_evidently(drift_report_filename, **kwargs):
    baseline_dict = kwargs['ti'].xcom_pull(key='baseline_df', task_ids='load_data')
    new_data_dict = kwargs['ti'].xcom_pull(key='new_data_df', task_ids='load_data')
    baseline_df = pd.DataFrame.from_dict(baseline_dict)
    baseline_df['datetime'] = pd.to_datetime(baseline_df['datetime'], errors='coerce')
    
    new_data_df = pd.DataFrame.from_dict(new_data_dict)
    new_data_df['datetime'] = pd.to_datetime(new_data_df['datetime'], errors='coerce')

    detector = DataDriftDetector(baseline_df, new_data_df)
    
    current_script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = current_script_dir + "/../reports/" + drift_report_filename

    evidently_results = detector.detect_drift_evidently(file_path)
    
    print("Evidently Drift Detection Results:", evidently_results)
    kwargs['ti'].xcom_push(key='evidently_results', value=evidently_results)

# Kolmogorov-Smirnov Test for drift detection
def detect_drift_ks_test(**kwargs):
    baseline_dict = kwargs['ti'].xcom_pull(key='baseline_df', task_ids='load_data')
    new_data_dict = kwargs['ti'].xcom_pull(key='new_data_df', task_ids='load_data')
    baseline_df = pd.DataFrame.from_dict(baseline_dict)
    new_data_df = pd.DataFrame.from_dict(new_data_dict)

    detector = DataDriftDetector(baseline_df, new_data_df)
    ks_test_results = detector.detect_drift_ks_test()
    print("KS Test Drift Results:", ks_test_results)
    kwargs['ti'].xcom_push(key='ks_test_results', value=ks_test_results)

# Population Stability Index Test for drift detection
def detect_drift_psi(**kwargs):
    baseline_dict = kwargs['ti'].xcom_pull(key='baseline_df', task_ids='load_data')
    new_data_dict = kwargs['ti'].xcom_pull(key='new_data_df', task_ids='load_data')
    baseline_df = pd.DataFrame.from_dict(baseline_dict)
    new_data_df = pd.DataFrame.from_dict(new_data_dict)

    detector = DataDriftDetector(baseline_df, new_data_df)
    psi_results = detector.detect_drift_psi()
    print("PSI Drift Results:", psi_results)
    kwargs['ti'].xcom_push(key='psi_results', value=psi_results)