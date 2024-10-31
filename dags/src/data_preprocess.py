import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from data_pipeline import get_data_from_dvc
import datetime
import subprocess

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
    df = pd.read_json(df_json)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    print("Data cleaning complete: missing values and duplicates removed.")

    json_data_cleaned = df.to_json(orient='records', lines=False)
    return json_data_cleaned

# Step 2: Feature Engineering
def engineer_features(df_json):
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

# Step 3: Add Cyclic Features
def add_cyclic_features(df_json):
    df = pd.read_json(df_json)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['month'] = df['datetime'].dt.month
    df['month_sin'] = np.round(np.sin(2 * np.pi * df['month'] / 12), decimals=6)
    df['month_cos'] = np.round(np.cos(2 * np.pi * df['month'] / 12), decimals=6)
    df.drop(columns=['month'], inplace=True)
    print("Cyclic features added for month seasonality.")

    json_data = df.to_json(orient='records', lines=False)
    return json_data

# Step 4: Normalize and Encode Data
def normalize_and_encode(df_json):
    df = pd.read_json(df_json)

    columns_to_normalize = df.select_dtypes(include=[np.number]).columns.difference(['month_sin', 'month_cos'])
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

# Step 5: Feature Selection
def select_final_features(df):
    df = pd.read_json(df_json)

    selected_features = [
        'precipMM', 'weatherCode', 'visibility', 'HeatIndexF', 'WindChillF',
        'windspeedMiles', 'FeelsLikeF', 'tempF_rolling_mean', 'windspeedMiles_rolling_mean',
        'humidity_rolling_mean', 'value', 'pressure', 'pressureInches', 'cloudcover', 'uvIndex',
        'tempF_rolling_std', 'windspeedMiles_rolling_std', 'humidity_rolling_std',
        'tempF_lag_2', 'windspeedMiles_lag_2', 'humidity_lag_2',
        'tempF_lag_4', 'windspeedMiles_lag_4', 'humidity_lag_4',
        'tempF_lag_6', 'windspeedMiles_lag_6', 'humidity_lag_6',
        'month_sin', 'month_cos'
    ]
    df_selected = df[selected_features]
    print("Feature selection complete: selected features retained.")

    json_data_selected = df_selected.to_json(orient='records', lines=False)
    return json_data_selected



