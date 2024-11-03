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
    # Get total number of rows in the file
    total_rows = sum(1 for _ in open(file_path)) - 1  # Exclude header row
    chunk_size = max(1, total_rows // 100)  # Ensure at least 1 row per chunk

    print(f"Total rows: {total_rows}, Chunk size: {chunk_size}")

    # Initialize the preprocessor object
    preprocess_obj = DataPreprocessor()
    
    # Define the output file path for the preprocessed data
    preprocessed_file_path = os.path.join(os.path.dirname(file_path), "data_preprocess.csv")

    # Open the output file in write mode to prepare for saving processed chunks
    with open(preprocessed_file_path, 'w') as output_file:
        # Read the data in chunks
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
            print(f"Processing chunk {i+1}...")
            
            # Ensure columns are in the correct data type
            chunk['datetime'] = chunk['datetime'].astype(str)
            chunk['zone'] = chunk['zone'].astype(str)
            chunk['subba-name'] = chunk['subba-name'].astype(str)
            
            # Convert to JSON format for DAG compatibility
            df_json = chunk.to_json(orient='records', lines=False)
            
            # Preprocessing steps
            df_json = preprocess_obj.clean_data(df_json)
            df_json = preprocess_obj.engineer_features(df_json)
            df_json = preprocess_obj.add_cyclic_features(df_json)
            df_json = preprocess_obj.normalize_and_encode(df_json)
            df_json = preprocess_obj.select_final_features(df_json)
            
            # Convert back to DataFrame
            processed_chunk = pd.read_json(df_json)
            
            # Save the chunk to CSV, appending from the second chunk onward
            header = (i == 0)  # Write header only for the first chunk
            processed_chunk.to_csv(output_file, index=False, mode='a', header=header)
            print(f"Chunk {i+1} processed and saved.")
    
    print("All chunks processed and saved to the final preprocessed file.")

    return preprocessed_file_path