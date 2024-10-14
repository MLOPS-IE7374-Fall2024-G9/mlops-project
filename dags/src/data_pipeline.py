from dataset.scripts.data_downloader import *
from dataset.scripts.dvc_manager import *

def update_dataset(start_date, end_date, today_flag=0):
    # function to fetch new data based on dates and region and update it on dvc, returns df
    region = ""
    local_save = 0

    data_df = update_and_save_data(start_date=start_date, end_date=end_date, today_flag=today_flag, region=region, local_save=local_save)
    return data_df

def upload_data_to_dvc(df):
    # call dvc manager function to upload the data back to dvc, returns none
    pass

def download_data_from_dvc():
    # call dvc manager function to download the data and get it in a dataframe, returns df
    pass

def preprocess_data(df):
    # preprocess the df to remove uncessary data points, returns a df
    pass



