import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from dags.src.data_preprocess import(
    clean_data, 
    engineer_features,
    add_cyclic_features,
    normalize_and_encode
)

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
    
    assert df_normalized['category'].dtype == int, "Category column not label-encoded as integer"
    
    unique_encoded_values = sorted(df_normalized['category'].unique())
    expected_labels = [0, 1, 2]  # Based on ["A", "B", "C"]
    assert unique_encoded_values == expected_labels, f"Unexpected label encoding: {unique_encoded_values}"


def main():

    test_clean_data()
    print("test_clean_data passed.") 

    test_engineer_features()
    print("test_engineer_features passed.")

    test_add_cyclic_features()
    print("test_add_cyclic_features passed.")

    test_normalize_and_encode()
    print("test_normalize_and_encode passed.")