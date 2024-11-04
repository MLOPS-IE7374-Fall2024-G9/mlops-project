import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import datetime
import subprocess
import os

class DataPreprocessor:
    """
    A class to preprocess data for a machine learning pipeline, including data cleaning,
    feature engineering, cyclic feature addition, normalization, encoding, and DVC tracking.
    """
    def __init__(self):
        pass
    
    def save_data(self, df, step_name="processed_data"):
        """
        Saves DataFrame to CSV and tracks with DVC for version control.
        """
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        filename = f"{step_name}_{date_str}.csv"
        
        try:
            df.to_csv(filename, index=False)
            print(f"Data saved locally as {filename}.")
            subprocess.run(["dvc", "add", filename], check=True)
            subprocess.run(["git", "add", f"{filename}.dvc"], check=True)
            subprocess.run(["git", "commit", "-m", f"Add {step_name} with DVC tracking"], check=True)
            subprocess.run(["git", "push", "origin", "main"], check=True)
            subprocess.run(["dvc", "push"], check=True)
            print(f"Data saved as {filename} and pushed to DVC.")
        except Exception as e:
            print(f"An error occurred: {e}")
        return filename

    def clean_data(self, df_json):
        """
        Cleans data by removing missing values and duplicates.
        """
        df = pd.read_json(df_json)
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        print("Data cleaning complete: missing values and duplicates removed.")
        json_data_cleaned = df.to_json(orient='records', lines=False)
        return json_data_cleaned

    def engineer_features(self, df_json):
        """
        Engineers rolling and lag features to capture temporal patterns.
        """
        df = pd.read_json(df_json)
        window_size = 6
        df['tempF_rolling_mean'] = df['tempF'].rolling(window=window_size).mean()
        df['tempF_rolling_std'] = df['tempF'].rolling(window=window_size).std()
        df['windspeedMiles_rolling_mean'] = df['windspeedMiles'].rolling(window=window_size).mean()
        df['windspeedMiles_rolling_std'] = df['windspeedMiles'].rolling(window=window_size).std()
        df['humidity_rolling_mean'] = df['humidity'].rolling(window=window_size).mean()
        df['humidity_rolling_std'] = df['humidity'].rolling(window=window_size).std()
        
        for lag in [2, 4, 6]:
            df[f'tempF_lag_{lag}'] = df['tempF'].shift(lag)
            df[f'windspeedMiles_lag_{lag}'] = df['windspeedMiles'].shift(lag)
            df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
        
        df.dropna(inplace=True)
        print("Feature engineering complete: rolling and lag features added.")
        json_data = df.to_json(orient='records', lines=False)
        return json_data

    def add_cyclic_features(self, df_json):
        """
        Adds cyclic features to capture seasonality patterns.
        """
        df = pd.read_json(df_json)
        df['datetime_1'] = pd.to_datetime(df['datetime'])
        df['month'] = df['datetime_1'].dt.month
        df['month_sin'] = np.round(np.sin(2 * np.pi * df['month'] / 12), decimals=6)
        df['month_cos'] = np.round(np.cos(2 * np.pi * df['month'] / 12), decimals=6)
        df.drop(columns=['month'], inplace=True)
        print("Cyclic features added for month seasonality.")
        json_data = df.to_json(orient='records', lines=False)
        return json_data

    def normalize_and_encode(self, df_json):
        """
        Normalizes numerical features and encodes categorical features.
        """
        df = pd.read_json(df_json)
        columns_to_normalize = df.select_dtypes(include=[np.number]).columns.difference(['month_sin', 'month_cos', "zone", "datetime", "subba-name"])
        df[columns_to_normalize] = df[columns_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        df['month_cos'] = (df['month_cos'] + 1) / 2
        df['month_sin'] = (df['month_sin'] + 1) / 2

        for col in df.select_dtypes(include=['object']).columns:
            if col != 'datetime':
                df[col] = df[col].astype(str)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])

        print("Data normalization and encoding complete.")
        json_data = df.to_json(orient='records', lines=False)
        return json_data

    def select_final_features(self, df_json):
        """
        Selects relevant features for the final dataset.
        """
        df = pd.read_json(df_json)
        selected_features = [
            'datetime', 'precipMM', 'weatherCode', 'visibility', 'HeatIndexF', 'WindChillF',
            'windspeedMiles', 'FeelsLikeF', 'tempF_rolling_mean', 'windspeedMiles_rolling_mean',
            'humidity_rolling_mean', 'value', 'pressure', 'pressureInches', 'cloudcover', 'uvIndex',
            'tempF_rolling_std', 'windspeedMiles_rolling_std', 'humidity_rolling_std',
            'tempF_lag_2', 'windspeedMiles_lag_2', 'humidity_lag_2',
            'tempF_lag_4', 'windspeedMiles_lag_4', 'humidity_lag_4',
            'tempF_lag_6', 'windspeedMiles_lag_6', 'humidity_lag_6',
            'month_sin', 'month_cos', 'subba-name', 'zone'
        ]
        df_selected = df[selected_features]
        print("Feature selection complete: selected features retained.")
        json_data_selected = df_selected.to_json(orient='records', lines=False)
        return json_data_selected
    
    def preprocess_pipeline(self, file_path, chunk_by_subba=True):
        """
        Processes the data by either chunking based on unique 'subba-name' values or by row count (100 rows).
        Each subset is processed independently and saved in a single output CSV file.
        """
        # Define the output file path for the preprocessed data
        preprocessed_file_path = os.path.join(os.path.dirname(file_path), "data_preprocess.csv")

        if chunk_by_subba:
            # Load the entire dataset
            data = pd.read_csv(file_path)
            unique_subbas = data['subba-name'].unique()  # Get unique values of 'subba-name'

            print(f"Total unique 'subba-name' values: {len(unique_subbas)}")

            with open(preprocessed_file_path, 'w') as output_file:
                for i, subba in enumerate(unique_subbas):
                    print(f"Processing chunk {i+1} for subba-name: {subba}...")
                    
                    # Filter the data for the current 'subba-name'
                    chunk = data[data['subba-name'] == subba]

                    # Ensure columns are in the correct data type
                    chunk['datetime'] = chunk['datetime'].astype(str)
                    chunk['zone'] = chunk['zone'].astype(str)
                    chunk['subba-name'] = chunk['subba-name'].astype(str)
                    
                    # Convert to JSON format for DAG compatibility
                    df_json = chunk.to_json(orient='records', lines=False)
                    
                    # Preprocessing steps
                    df_json = self.clean_data(df_json)
                    df_json = self.engineer_features(df_json)
                    df_json = self.add_cyclic_features(df_json)
                    df_json = self.normalize_and_encode(df_json)
                    df_json = self.select_final_features(df_json)
                    
                    # Convert back to DataFrame
                    processed_chunk = pd.read_json(df_json)
                    
                    # Save the chunk to CSV, appending from the second chunk onward
                    header = (i == 0)  # Write header only for the first chunk
                    processed_chunk.to_csv(output_file, index=False, mode='a', header=header)
                    print(f"Chunk {i+1} processed and saved for subba-name: {subba}.")
        else:
            # Get total number of rows in the file
            total_rows = sum(1 for _ in open(file_path)) - 1  # Exclude header row
            chunk_size = max(1, total_rows // 100)  # Ensure at least 1 row per chunk
            print(f"Total rows: {total_rows}, Chunk size: {chunk_size}")

            with open(preprocessed_file_path, 'w') as output_file:
                for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                    print(f"Processing chunk {i+1}...")

                    # Ensure columns are in the correct data type
                    chunk['datetime'] = chunk['datetime'].astype(str)
                    chunk['zone'] = chunk['zone'].astype(str)
                    chunk['subba-name'] = chunk['subba-name'].astype(str)
                    
                    # Convert to JSON format for DAG compatibility
                    df_json = chunk.to_json(orient='records', lines=False)
                    
                    # Preprocessing steps
                    df_json = self.clean_data(df_json)
                    df_json = self.engineer_features(df_json)
                    df_json = self.add_cyclic_features(df_json)
                    df_json = self.normalize_and_encode(df_json)
                    df_json = self.select_final_features(df_json)
                    
                    # Convert back to DataFrame
                    processed_chunk = pd.read_json(df_json)
                    
                    # Save the chunk to CSV, appending from the second chunk onward
                    header = (i == 0)  # Write header only for the first chunk
                    processed_chunk.to_csv(output_file, index=False, mode='a', header=header)
                    print(f"Chunk {i+1} processed and saved.")
            
        print("All chunks processed and saved to the final preprocessed file.")
        return preprocessed_file_path
