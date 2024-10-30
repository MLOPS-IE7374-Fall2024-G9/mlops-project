import os
import sys

from dataset.scripts.dvc_manager import *
from dataset.scripts.data import *
from dataset.scripts.data_schema import *

# ----------------------------------------------------------
# DataCollector
def get_start_end_dates() -> tuple[str, str]:
    data_obj = DataCollector()
    yesterday, today = data_obj.get_yesterday_dates()
    return yesterday, today

def get_last_k_start_end_dates(days: int) -> tuple[str, str]:
    data_obj = DataCollector()
    _, today = data_obj.get_yesterday_dates()
    
    # Calculate the start date based on the specified number of days
    start_date = (pd.to_datetime(today) - timedelta(days=days-1)).strftime('%d-%m-%Y')
    
    return start_date, today

def get_updated_data_from_api(dates: tuple[str, str]) -> pd.DataFrame:
    data_obj = DataCollector()
    data_regions = DataRegions()

    start_date, end_date = dates
    updated_data = data_obj.get_data_from_api(data_regions.regions, start_date, end_date, today_flag=0)
    
    json_data = updated_data.to_json(orient='records', lines=False)
    return json_data

def merge_data(api_json, dvc_json):
    api_df = pd.read_json(api_json)
    dvc_df = pd.read_json(dvc_json)

    if dvc_df is not None:
        updated_data_df = pd.concat([dvc_df, api_df], ignore_index=True)
    else:
        updated_data_df = api_df

    json_data = updated_data_df.to_json(orient='records', lines=False)
    return json_data


def redundant_removal(data_json):
    data_df = pd.read_json(data_json)

    # Remove duplicate rows based on the 'datetime' column
    data_df = data_df.drop_duplicates(subset='datetime')
    
    json_data = data_df.to_json(orient='records', lines=False)
    return json_data


# ----------------------------------------------------------
# DVC Manager
def get_data_from_dvc():
    dvc_manager_obj = DVCManager()
    df = dvc_manager_obj.download_data_from_dvc()
    
    json_data = df.to_json(orient='records', lines=False)
    return json_data

def update_data_to_dvc(df_json: dict) -> None:
    dvc_manager_obj = DVCManager()
    df = pd.read_json(df_json)
    dvc_manager_obj.upload_data_to_dvc(df, dvc_manager_obj.all_data_filename)

# ----------------------------------------------------------
# Data Schema
def get_statistics_and_infer(df):
    schema_stats_generator = DataSchemaAndStatistics(df)
    stats = schema_stats_generator.generate_statistics()
    schema = schema_stats_generator.infer_schema()
    schema_stats_generator.save_schema("dataset/")
    return stats

def infer_schema(df):
    schema_stats_generator = DataSchemaAndStatistics(df)
    

def validate_data(new_df):
    schema_stats_generator = DataSchemaAndStatistics(df)
    anomalies = schema_stats_generator.validate_data(new_df)

# ----------------------------------------------------------
# Data Bias


# ----------------------------------------------------------