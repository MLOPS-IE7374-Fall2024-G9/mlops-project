import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import datetime
import subprocess

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dataset.scripts.data_preprocess import *

# Function to Save Data to CSV and Track with DVC, Including Date in Filename
def save_data(df, step_name="processed_data"):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"{step_name}_{date_str}.csv"
    
    try:
        # Save DataFrame to CSV
        df.to_csv(filename, index=False)
        print(f"Data saved locally as {filename}.")

        # DVC tracking
        subprocess.run(["dvc", "add", filename], check=True)
        subprocess.run(["git", "add", f"{filename}.dvc"], check=True)
        subprocess.run(["git", "commit", "-m", f"Add {step_name} with DVC tracking"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        subprocess.run(["dvc", "push"], check=True)

        print(f"Data saved as {filename} and pushed to DVC.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return filename

# Step 1: Data Cleaning
def clean_data(df_json):
    preprocess_obj = DataPreprocessor()
    json_data_cleaned = preprocess_obj.clean_data(df_json)
    return json_data_cleaned

# Step 2: Feature Engineering
def engineer_features(df_json):
    preprocess_obj = DataPreprocessor()
    json_data = preprocess_obj.engineer_features(df_json)
    return json_data

# Step 3: Add Cyclic Features
def add_cyclic_features(df_json):
    preprocess_obj = DataPreprocessor()
    json_data = preprocess_obj.add_cyclic_features(df_json)
    return json_data

# Step 4: Normalize and Encode Data
def normalize_and_encode(df_json):
    preprocess_obj = DataPreprocessor()
    json_data = preprocess_obj.normalize_and_encode(df_json)
    return json_data

# Step 5: Feature Selection
def select_final_features(df_json):
    preprocess_obj = DataPreprocessor()
    json_data_selected = preprocess_obj.select_final_features(df_json)
    return json_data_selected


def preprocess_pipeline(file_path):
    # Initialize the preprocessor object
    preprocess_obj = DataPreprocessor()
    preprocessed_file_path = preprocess_obj.preprocess_pipeline(file_path, chunk_by_subba=False)
    return preprocessed_file_path
