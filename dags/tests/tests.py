import sys
import os

# Adjust the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytest
import json
from io import StringIO
import warnings

from src.data_download import * 
from dags.src.data_preprocess import *
from dags.src.data_schema_validation import *
from dags.src.data_bias_detection_and_mitigation import detect_bias, conditional_mitigation
from dags.src.data_drift import DataDriftDetector
import os
import shutil
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from unittest.mock import patch, MagicMock
from dags.src.model_bias_detection_and_mitigation import (setup_local_run_folder, load_splits, upload_to_gcp, generate_metric_frame, 
                    save_metric_frame, generate_and_save_bias_analysis, identify_bias_by_threshold)


# Ignore all warnings in tests
warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# Sample data for testing
sample_csv_data = """
datetime,precipMM,weatherCode,visibility,HeatIndexF,WindChillF,windspeedMiles,FeelsLikeF,value
2019-01-01T00,0.5,100,10,36,29,5,32,1500
2019-01-01T01,0.2,200,12,37,30,7,33,1600
"""

sample_api_json = """
[
    {"datetime": "2019-01-01T00", "precipMM": 0.5, "weatherCode": 100, "visibility": 10, "HeatIndexF": 36, "WindChillF": 29, "windspeedMiles": 5, "FeelsLikeF": 32, "value": 1500},
    {"datetime": "2019-01-01T01", "precipMM": 0.2, "weatherCode": 200, "visibility": 12, "HeatIndexF": 37, "WindChillF": 30, "windspeedMiles": 7, "FeelsLikeF": 33, "value": 1600}
]
"""

# ----------------------------------------------------------
# data_download.py
def test_get_start_end_dates():
    today = datetime.datetime.now().strftime('%d-%m-%Y')
    yesterday = (datetime.datetime.now() - timedelta(days=1)).strftime('%d-%m-%Y')
    func_yesterday, func_today = get_start_end_dates()
    
    assert func_yesterday == yesterday, f"Expected {yesterday} but got {func_yesterday}"
    assert func_today == today, f"Expected {today} but got {func_today}"

def test_get_last_k_start_end_dates():
    days = 5
    today = datetime.datetime.now().strftime('%d-%m-%Y')
    start_date = (datetime.datetime.now() - timedelta(days=days-1)).strftime('%d-%m-%Y')
    func_start_date, func_today = get_last_k_start_end_dates(days)
    
    assert func_start_date == start_date, f"Expected {start_date} but got {func_start_date}"
    assert func_today == today, f"Expected {today} but got {func_today}"

def test_get_updated_data_from_api():
    func_yesterday, func_today = get_start_end_dates()
    dates = (func_yesterday, func_today)

    api_json = get_updated_data_from_api(dates)
    api_df = pd.read_json(api_json, orient='records')

    func_yesterday_formatted = datetime.datetime.strptime(func_yesterday, "%d-%m-%Y").strftime("%Y-%m-%d")
    
    assert any(func_yesterday_formatted in str(dt) for dt in api_df['datetime'].values), f"Expected yesterday's date ({func_yesterday_formatted}) in API data but it was not found."


# ----------------------------------------------------------
# data_preprocess.py
def test_clean_data():
    df_json = pd.DataFrame({
        "col1": [1, 2, None, 3, 1],
        "col2": [None, 5, 6, 5, 4]
    }).to_json(orient='records', lines=False)
    
    cleaned_json = clean_data(df_json)
    df_cleaned = pd.read_json(cleaned_json)
    
    assert df_cleaned.isnull().sum().sum() == 0, "Missing values were not removed"
    assert len(df_cleaned) == 3, f"Expected 3 rows after removing duplicates, but got {len(df_cleaned)}"

def test_engineer_features():
    df_json = pd.DataFrame({
        "tempF": [70, 71, 69, 68, 72, 74, 73, 75, 76, 78],
        "windspeedMiles": [10, 12, 11, 13, 9, 14, 10, 15, 13, 11],
        "humidity": [50, 55, 53, 57, 52, 56, 54, 58, 59, 60]
    }).to_json(orient='records', lines=False)
    
    engineered_json = engineer_features(df_json)
    df_engineered = pd.read_json(engineered_json)
    
    rolling_columns = [
        'tempF_rolling_mean', 'tempF_rolling_std', 
        'windspeedMiles_rolling_mean', 'windspeedMiles_rolling_std', 
        'humidity_rolling_mean', 'humidity_rolling_std'
    ]
    for col in rolling_columns:
        assert col in df_engineered.columns, f"Expected column {col} is missing after feature engineering"
    
    lag_columns = [
        'tempF_lag_2', 'tempF_lag_4', 'tempF_lag_6',
        'windspeedMiles_lag_2', 'windspeedMiles_lag_4', 'windspeedMiles_lag_6',
        'humidity_lag_2', 'humidity_lag_4', 'humidity_lag_6'
    ]
    for col in lag_columns:
        assert col in df_engineered.columns, f"Expected lag column {col} is missing after feature engineering"
    
    assert df_engineered.isnull().sum().sum() == 0, "There are NaN values in the DataFrame after dropping rows"

    expected_rows = len(pd.read_json(df_json)) - max(6, 6)  
    assert len(df_engineered) == expected_rows, f"Expected {expected_rows} rows, but got {len(df_engineered)}"

def test_add_cyclic_features():
    df_json = pd.DataFrame({
        "datetime": [
            "2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01",
            "2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01",
            "2024-09-01", "2024-10-01", "2024-11-01", "2024-12-01"
        ]
    }).to_json(orient='records', lines=False)

    cyclic_json = add_cyclic_features(df_json)
    df_cyclic = pd.read_json(cyclic_json)
    
    assert "month_sin" in df_cyclic.columns, "Expected 'month_sin' column is missing"
    assert "month_cos" in df_cyclic.columns, "Expected 'month_cos' column is missing"
    
    assert df_cyclic['month_sin'].between(-1, 1).all(), "'month_sin' values are out of range [-1, 1]"
    assert df_cyclic['month_cos'].between(-1, 1).all(), "'month_cos' values are out of range [-1, 1]"
    
    expected_sin_cos = {
        "2024-01-01": (0.5, 0.866025),  # Jan: (sin, cos)
        "2024-07-01": (-0.5, -0.866025) # Jul: (sin, cos)
    }
    
    for date_str, (expected_sin, expected_cos) in expected_sin_cos.items():
        row = df_cyclic[df_cyclic['datetime'] == date_str]
        assert np.isclose(row['month_sin'].values[0], expected_sin, atol=1e-5), f"Incorrect 'month_sin' for {date_str}"
        assert np.isclose(row['month_cos'].values[0], expected_cos, atol=1e-5), f"Incorrect 'month_cos' for {date_str}"

def test_normalize_and_encode():
    df_json = pd.DataFrame({
        "tempF": [60, 70, 80, 90],
        "humidity": [30, 40, 50, 60],
        "windspeedMiles": [5, 10, 15, 20],
        "month_sin": [0.5, 0.866025, -0.5, -0.866025], 
        "month_cos": [0.866025, 0.5, -0.866025, -0.5],  
        "category": ["A", "B", "A", "C"], 
        "datetime": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
    }).to_json(orient='records', lines=False)
    
    normalized_json = normalize_and_encode(df_json)
    df_normalized = pd.read_json(normalized_json)
    
    columns_to_normalize = ["tempF", "humidity", "windspeedMiles"]
    for col in columns_to_normalize:
        assert df_normalized[col].between(0, 1).all(), f"{col} values not normalized to [0, 1]"

    assert df_normalized['month_sin'].between(0, 1).all(), "'month_sin' values are out of range [0, 1]"
    assert df_normalized['month_cos'].between(0, 1).all(), "'month_cos' values are out of range [0, 1]"

    print(df_normalized['category'].dtype)
    print(df_normalized['category'])
    
    assert df_normalized['category'].dtype == int, "Category column not label-encoded as integer"
    
    unique_encoded_values = sorted(df_normalized['category'].unique())
    expected_labels = [0, 1, 2]  # Based on ["A", "B", "C"]
    assert unique_encoded_values == expected_labels, f"Unexpected label encoding: {unique_encoded_values}"
    
# ---------------------------------------------------------------
# data_schema.py
@pytest.fixture
def sample_csv_file(tmpdir):
    file_path = tmpdir.join("sample_data.csv")
    with open(file_path, "w") as f:
        f.write(sample_csv_data)
    return str(file_path)

@pytest.fixture
def sample_api_json_data():
    return sample_api_json

def test_validate_data(sample_csv_file, sample_api_json_data):
    # Check if API data validates correctly against CSV schema
    is_valid = validate_data(sample_csv_file, sample_api_json_data)
    assert is_valid == 1, "API data did not validate against schema."

def test_fix_anomalies(sample_api_json_data):
    # Introduce negative and NaN anomalies in data
    api_data = json.loads(sample_api_json_data)
    api_data[0]["visibility"] = -5
    api_data[1]["HeatIndexF"] = None

    # Fix anomalies
    fixed_data = fix_anomalies(json.dumps(api_data))
    fixed_df = pd.read_json(StringIO(fixed_data))

    # Check for fixed anomalies
    assert (fixed_df["visibility"] >= 0).all(), "Negative values in 'visibility' not fixed"
    assert fixed_df["HeatIndexF"].isnull().sum() == 0, "NaN values in 'HeatIndexF' not fixed"

# ---------------------------------------------------------------
# data_bias_detection.py 

def test_detect_bias_output_structure():
    # Create a sample DataFrame for testing bias detection
    data = pd.DataFrame({
        'value': [1, 0, 1, np.nan, 1, 0, 1, np.nan, 0, 1, 1, 0],
        'subba-name': ['A', 'B', 'A', 'A', 'B', 'B', 'A', 'B', 'A', 'A', 'B', 'B']
    })
    target_col = 'value'
    sensitive_col = 'subba-name'

    # Run the detect_bias function
    bias_output = detect_bias(data, target_col, sensitive_col)

    # Assertions to check the structure of the output
    assert isinstance(bias_output, dict), "Expected output to be a dictionary"
    assert 'metrics_by_group' in bias_output, "Expected 'metrics_by_group' key in output"
    assert 'overall_metrics' in bias_output, "Expected 'overall_metrics' key in output"
    assert 'demographic_parity_difference' in bias_output, "Expected 'demographic_parity_difference' key in output"

def test_detect_bias_metrics_values():
    # Create a sample DataFrame for testing
    data = pd.DataFrame({
        'value': [1, 0, 1, np.nan, 1, 0, 1, np.nan, 0, 1, 1, 0],
        'subba-name': ['A', 'B', 'A', 'A', 'B', 'B', 'A', 'B', 'A', 'A', 'B', 'B']
    })
    target_col = 'value'
    sensitive_col = 'subba-name'

    # Run the detect_bias function
    bias_output = detect_bias(data, target_col, sensitive_col)
    metrics_by_group = bias_output['metrics_by_group']
    overall_metrics = bias_output['overall_metrics']

    # Assertions to check the validity of the metrics
    assert not metrics_by_group.empty, "Metrics by group should not be empty"
    assert overall_metrics['Selection Rate'] >= 0, "Selection rate should be non-negative"
    assert overall_metrics['Selection Rate'] <= 1, "Selection rate should be at most 1"

def test_conditional_mitigation_output():
    # Create a sample DataFrame for testing conditional mitigation
    data = pd.DataFrame({
        'value': [1, 0, 1, np.nan, 1, 0, 1, np.nan, 0, 1],
        'subba-name': ['A', 'B', 'A', 'A', 'B', 'B', 'A', 'B', 'A', 'A']
    })
    target_col = 'value'
    sensitive_col = 'subba-name'

    # Run detect_bias and conditional_mitigation functions
    bias_output = detect_bias(data, target_col, sensitive_col)
    mitigated_data = conditional_mitigation(data, target_col, sensitive_col, bias_output)

    # Assertions to check the structure of the mitigated data
    assert isinstance(mitigated_data, pd.DataFrame), "Expected mitigated data to be a DataFrame"
    assert len(mitigated_data) >= len(data), "Mitigated data should have more rows than original data due to resampling"

def test_conditional_mitigation_groups():
    # Create a sample DataFrame for testing group mitigation
    data = pd.DataFrame({
        'value': [1, 0, 1, np.nan, 1, 0, 1, np.nan, 0, 1, 1, 0],
        'subba-name': ['A', 'B', 'A', 'A', 'B', 'B', 'A', 'B', 'A', 'A', 'B', 'B']
    })
    target_col = 'value'
    sensitive_col = 'subba-name'

    # Run detect_bias and conditional_mitigation functions
    bias_output = detect_bias(data, target_col, sensitive_col)
    mitigated_data = conditional_mitigation(data, target_col, sensitive_col, bias_output)

    # Check if expected subgroups are present in the mitigated data
    unique_groups = mitigated_data[sensitive_col].unique()
    assert 'A' in unique_groups, "Expected group 'A' to be in mitigated data"
    assert 'B' in unique_groups, "Expected group 'B' to be in mitigated data"



# Setup test data and environment
def setup_environment():
    temp_dir = "temp_test_dir"
    os.makedirs(temp_dir, exist_ok=True)
    X_train, y_train = make_regression(n_samples=100, n_features=10, noise=0.1)
    X_test, y_test = make_regression(n_samples=20, n_features=10, noise=0.1)
    sensitive_train = pd.Series(["group1" if i % 2 == 0 else "group2" for i in range(100)])
    sensitive_test = pd.Series(["group1" if i % 2 == 0 else "group2" for i in range(20)])
    
    pd.DataFrame(X_train).to_csv(os.path.join(temp_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(temp_dir, "X_test.csv"), index=False)
    pd.Series(y_train).to_csv(os.path.join(temp_dir, "y_train.csv"), index=False)
    pd.Series(y_test).to_csv(os.path.join(temp_dir, "y_test.csv"), index=False)
    sensitive_train.to_csv(os.path.join(temp_dir, "sensitive_train.csv"), index=False)
    sensitive_test.to_csv(os.path.join(temp_dir, "sensitive_test.csv"), index=False)
    return temp_dir, X_train, X_test, y_train, y_test, sensitive_train, sensitive_test

# Clean up test environment
def cleanup_environment(temp_dir):
    shutil.rmtree(temp_dir)

# Test setup_local_run_folder
def test_setup_local_run_folder():
    temp_dir, *_ = setup_environment()
    local_path, gcp_path = setup_local_run_folder(temp_dir, "test_run")
    assert os.path.exists(local_path), "Local path should be created."
    assert "test_run" in gcp_path, "GCP path should contain the run type."
    shutil.rmtree(local_path)
    cleanup_environment(temp_dir)

# Test load_splits
def test_load_splits():
    temp_dir, *_ = setup_environment()
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = load_splits(
        os.path.join(temp_dir, "X_train.csv"),
        os.path.join(temp_dir, "X_test.csv"),
        os.path.join(temp_dir, "y_train.csv"),
        os.path.join(temp_dir, "y_test.csv"),
        os.path.join(temp_dir, "sensitive_train.csv"),
        os.path.join(temp_dir, "sensitive_test.csv"),
    )
    assert len(X_train) == 100, "X_train should have 100 rows."
    assert len(X_test) == 20, "X_test should have 20 rows."
    assert len(sensitive_train) == 100, "sensitive_train should have 100 entries."
    cleanup_environment(temp_dir)

# Test upload_to_gcp
@patch("script.storage.Client")
def test_upload_to_gcp(mock_client):
    temp_dir, *_ = setup_environment()
    mock_bucket = MagicMock()
    mock_client.return_value.get_bucket.return_value = mock_bucket

    test_file_path = os.path.join(temp_dir, "test_file.txt")
    with open(test_file_path, "w") as f:
        f.write("test content")

    upload_to_gcp(temp_dir, "gcp_folder", "test_bucket")
    mock_bucket.blob.assert_called_with("gcp_folder/test_file.txt")
    mock_bucket.blob().upload_from_filename.assert_called_with(test_file_path)
    os.remove(test_file_path)
    cleanup_environment(temp_dir)

# Test generate_metric_frame
def test_generate_metric_frame():
    temp_dir, X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = setup_environment()
    model = LinearRegression().fit(X_train, y_train)
    metric_frame = generate_metric_frame(model, X_test, y_test, sensitive_test)
    assert "mae" in metric_frame.overall, "MetricFrame should contain MAE."
    cleanup_environment(temp_dir)

# Test save_metric_frame
@patch("script.upload_to_gcp")
def test_save_metric_frame(mock_upload):
    temp_dir, X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = setup_environment()
    model = LinearRegression().fit(X_train, y_train)
    metric_frame = generate_metric_frame(model, X_test, y_test, sensitive_test)

    save_metric_frame(metric_frame, "test_model", "test_bucket", temp_dir)
    mock_upload.assert_called()
    cleanup_environment(temp_dir)

# Test generate_and_save_bias_analysis
def test_generate_and_save_bias_analysis():
    temp_dir, X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = setup_environment()
    model = LinearRegression().fit(X_train, y_train)
    metric_frame = generate_metric_frame(model, X_test, y_test, sensitive_test)

    graph_path = generate_and_save_bias_analysis(metric_frame, "mae", temp_dir)
    assert os.path.exists(graph_path), "Bias analysis graph should be saved."
    os.remove(graph_path)
    cleanup_environment(temp_dir)

# Test identify_bias_by_threshold
def test_identify_bias_by_threshold():
    temp_dir, X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = setup_environment()
    model = LinearRegression().fit(X_train, y_train)
    metric_frame = generate_metric_frame(model, X_test, y_test, sensitive_test)

    biased_flag, biased_groups, metric_ratio = identify_bias_by_threshold(metric_frame, "mae")
    assert isinstance(biased_flag, bool), "biased_flag should be a boolean."
    assert isinstance(biased_groups, pd.DataFrame), "biased_groups should be a DataFrame."
    cleanup_environment(temp_dir)




# ---------------------------------------------------------------
# data_drift.py 
# ---------------------------------------------------------------

# Function to generate sample data with drift
def sample_data():
    baseline_data = pd.DataFrame({
        'tempF': np.random.normal(70, 10, 1000),
        'humidity': np.random.normal(50, 5, 1000),
        'pressure': np.random.normal(1013, 10, 1000)
    })
    new_data = pd.DataFrame({
        'tempF': np.random.normal(75, 10, 1000),  # Changing mean to simulate drift
        'humidity': np.random.normal(55, 5, 1000),
        'pressure': np.random.normal(1015, 10, 1000)
    })
    return baseline_data, new_data

baseline_df, new_data_df = sample_data()

# detect_drift_evidently Function
def test_detect_drift_evidently():
    detector = DataDriftDetector(baseline_df, new_data_df)
    drift_report_filename = "drift_report.json"
    evidently_results = detector.detect_drift_evidently(drift_report_filename)

    # Validate the results
    assert evidently_results is not None, "Evidently drift detection results should not be None"
    assert isinstance(evidently_results, dict), "Evidently drift detection results should be a dictionary"
   
    # Access the drift results for each column
    drift_by_columns = evidently_results["metrics"][1]["result"]["drift_by_columns"]
    
    # Check that each feature in the dataframe has a result entry
    for feature in baseline_df.columns:
        assert feature in drift_by_columns, f"Evidently results should contain drift data for feature '{feature}'"
        assert "drift_score" in drift_by_columns[feature], f"'{feature}' should have a 'drift_score' in the results"
        assert "drift_detected" in drift_by_columns[feature], f"'{feature}' should have a 'drift_detected' indicator in the results"


# detect_drift_ks_test Function
def test_detect_drift_ks_test():
    detector = DataDriftDetector(baseline_df, new_data_df)
    ks_test_results = detector.detect_drift_ks_test()

    # Validate the results
    assert ks_test_results is not None, "KS Test drift detection results should not be None"
    assert isinstance(ks_test_results, dict), "KS Test drift detection results should be a dictionary"
    # Check that each feature in the dataframe has a p_value in the results
    for feature, result in ks_test_results.items():
        assert "p_value" in result, f"KS Test results for feature '{feature}' should contain 'p_value'"
        assert 0 <= result["p_value"] <= 1, f"KS Test p-value for feature '{feature}' should be within [0, 1]"

# detect_drift_psi Function
def test_detect_drift_psi():
    detector = DataDriftDetector(baseline_df, new_data_df)
    psi_results = detector.detect_drift_psi()

    # Validate the results
    assert psi_results is not None, "PSI drift detection results should not be None"
    assert isinstance(psi_results, dict), "PSI drift detection results should be a dictionary"
    # Check that each feature in the dataframe has a PSI in the results
    for feature, result in psi_results.items():
        assert "PSI" in result, f"PSI results for feature '{feature}' should contain 'PSI'"
        assert result["PSI"] >= 0, f"PSI value for feature '{feature}' should be non-negative"

# Run tests
if __name__ == "__main__":
    test_setup_local_run_folder()
    test_load_splits()
    test_upload_to_gcp()
    test_generate_metric_frame()
    test_save_metric_frame()
    test_generate_and_save_bias_analysis()
    test_identify_bias_by_threshold()
    print("All tests passed.")