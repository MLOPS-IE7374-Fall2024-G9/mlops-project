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


def main():

    test_clean_data()
    print("test_clean_data passed.") 
