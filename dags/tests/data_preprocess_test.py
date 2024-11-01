import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


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

def main():

    test_clean_data()
    print("test_clean_data passed.") 

    test_engineer_features()
    print("test_engineer_features passed.")
