import warnings
from data_preprocess import *
import os
import sys
import pandas as pd

# Suppress all warnings
warnings.filterwarnings("ignore")

def raw_to_preprocess(raw_file_path):
    """
    Preprocesses a large raw CSV file by splitting it into smaller chunks, 
    performing data cleaning and feature engineering on each chunk, 
    and merging them into a single preprocessed CSV file.
    """
    # Get total number of rows in the file
    total_rows = sum(1 for _ in open(raw_file_path)) - 1  # Exclude header row
    chunk_size = max(1, total_rows // 100)  # Ensure at least 1 row per chunk

    print(f"Total rows: {total_rows}, Chunk size: {chunk_size}")

    # Initialize the preprocessor object
    preprocess_obj = DataPreprocessor()
    
    # Define the output file path for the preprocessed data
    preprocessed_file_path = os.path.join(os.path.dirname(raw_file_path), "data_preprocess.csv")

    # Open the output file in write mode to prepare for saving processed chunks
    with open(preprocessed_file_path, 'w') as output_file:
        # Read the data in chunks
        for i, chunk in enumerate(pd.read_csv(raw_file_path, chunksize=chunk_size)):
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

if __name__ == "__main__":
    # Check if file path is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python data_preprocess_script.py <raw_file_path>")
    else:
        raw_to_preprocess(sys.argv[1])

# Usage:
# python data_preprocess_script.py /data/data_raw.csv
