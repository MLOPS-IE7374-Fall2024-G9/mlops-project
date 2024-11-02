import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dataset.scripts.dvc_manager import *
from dataset.scripts.data import *

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
    start_date = (datetime.now() - timedelta(days=days-1)).strftime('%d-%m-%Y')
    return start_date, today

def get_updated_data_from_api(dates: tuple[str, str]) -> pd.DataFrame:
    data_obj = DataCollector()
    data_regions = DataRegions()

    start_date, end_date = dates
    updated_data = data_obj.get_data_from_api(data_regions.regions, start_date, end_date, today_flag=0)
    
    json_data = updated_data.to_json(orient='records', lines=False)
    return json_data

def merge_data(api_json, dvc_file_path):
    api_df = pd.read_json(api_json)
    dvc_df = pd.read_csv(dvc_file_path)

    if dvc_df is not None:
        updated_data_df = pd.concat([dvc_df, api_df], ignore_index=True)
    else:
        updated_data_df = api_df
    
    save_df_to_csv(updated_data_df, dvc_file_path)
    return dvc_file_path

def redundant_removal(data_path):
    data_df = pd.read_csv(data_path)

    # Remove duplicate rows based on the 'datetime' column
    data_df = data_df.drop_duplicates(subset='datetime')
    
    save_df_to_csv(data_df, data_path)
    return data_path

# ----------------------------------------------------------
# DVC Manager
def get_data_from_dvc(filename):
    dvc_manager_obj = DVCManager()
    df, file_path = dvc_manager_obj.download_data_from_dvc(filename, save_local=1)
    
    # json_data = df.to_json(orient='records', lines=False)
    return file_path

def update_data_to_dvc(filename):
    dvc_manager_obj = DVCManager()
    df = pd.read_csv(filename)
    dvc_manager_obj.upload_data_to_dvc(df, filename)

def delete_local_dvc_data():
    dvc_manager_obj = DVCManager()
    dvc_manager_obj.delete_local_data()


# ----------------------------------------------------------
# utils 
def save_df_to_csv(df, filename):
    df.to_csv(filename, index=False)