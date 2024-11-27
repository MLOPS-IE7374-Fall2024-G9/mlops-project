import os 
import joblib
import warnings
import pandas as pd
import numpy as np
import argparse
import json

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
warnings.filterwarnings("ignore")

from dataset.scripts.data_preprocess import DataPreprocessor
from dataset.scripts.data import DataCollector
from mlflow_model_registry import MLflowModelRegistry

class ModelInference:
    def __init__(self, window_size=6):
        self.model_path = os.path.join(os.path.dirname(__file__), '../pickle/model.pkl')

        self.feature_columns = ["precipMM", "weatherCode", "visibility", "HeatIndexF", "WindChillF",
                                "windspeedMiles", "FeelsLikeF", "tempF_rolling_mean", "windspeedMiles_rolling_mean",
                                "humidity_rolling_mean", "pressure", "pressureInches", "cloudcover", "uvIndex",
                                "tempF_rolling_std", "windspeedMiles_rolling_std", "humidity_rolling_std",
                                "tempF_lag_2", "windspeedMiles_lag_2", "humidity_lag_2",
                                "tempF_lag_4", "windspeedMiles_lag_4", "humidity_lag_4",
                                "tempF_lag_6", "windspeedMiles_lag_6", "humidity_lag_6",
                                "month_sin", "month_cos"]
        
        self.value_column = "value"
        self.model = None
        self.window_size = window_size

        self.data_obj = DataCollector()
        self.data_preprocess_obj = DataPreprocessor()
        self.mlflow_registry = MLflowModelRegistry()

        path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.json'))
        with open(path, "r") as config_file:
            self.config = json.load(config_file)

    def download_model(self):
        self.mlflow_registry.fetch_and_initialize_latest_model(self.config.get("experimentation_name"))

    def load_model(self):
        self.model = joblib.load(self.model_path)

    def get_weather_data(self, location):
        # get date
        yesterday, today = self.data_obj.get_yesterday_dates()

        # get weather data
        data_df = self.data_obj.get_weather_data(location, yesterday, today)

        # process hourly
        data_df = self.data_obj.process_weather_data(data_df)
        data_df.rename(columns={"datetime": "datetime"}, inplace=True)
        data_df.rename(columns={"period": "datetime"}, inplace=True)

        # split based on window size
        data_df = data_df.tail(self.window_size + 1)
        return data_df
    
    def preprocess_input(self, input_df):
        # normalize
        input_df['datetime'] = input_df['datetime'].astype(str)

        df_json =  input_df.to_json(orient='records', lines=False)

        df_json = self.data_preprocess_obj.clean_data(df_json)
        df_json = self.data_preprocess_obj.engineer_features(df_json)
        df_json = self.data_preprocess_obj.add_cyclic_features(df_json)
        df_json = self.data_preprocess_obj.normalize_data_single(df_json)
        input_df_normalized = pd.read_json(df_json)

        # select features
        input_df_preprocessed = input_df_normalized[self.feature_columns]

        return input_df_preprocessed

    def denormalize_output(self, output):
        return self.data_preprocess_obj.denormalize_output(output)

    def predict(self, input_df):
        # preprocess input
        input_normalized = self.preprocess_input(input_df)

        # predict
        output = self.model.predict(input_normalized)

        # denormalize output
        output_denormalized = self.denormalize_output(output)

        return output_denormalized[0]
    

def main():
    parser = argparse.ArgumentParser(description="Run model inference.")
    parser.add_argument(
        "--coordinates",
        type=str,
        required=True,
        help="Coordinates of the location (latitude,longitude). Example: '42.3601,71.0589'",
    )

    args = parser.parse_args()

    # Initialize the model inference
    model_inference = ModelInference()

    try:
        # Load the model
        print("Loading model...")
        model_inference.load_model()

        # Get weather data based on location
        print(f"Fetching weather data for location: {args.coordinates}")
        input_df = model_inference.get_weather_data(location=args.coordinates)

        # Predict
        print("Running prediction...")
        predictions = model_inference.predict(input_df)

        print(f"Prediction: {predictions}")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # test
    # coordinates = "42.3601, 71.0589"
    # model_inference = ModelInference()
    # input_df = model_inference.get_weather_data(location=coordinates)
    # model_inference.load_model()
    # predictions = model_inference.predict(input_df)
    # print(predictions)
    main()


# usage - python model/scripts/inference.py --coordinates '42.3601,71.0589'