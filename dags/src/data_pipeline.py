from dataset.scripts.dvc_manager import *
from dataset.scripts.data_downloader import *
from datetime import datetime, timedelta


def get_start_end_dates() -> tuple[str, str]:
    """
    Get the start and end dates as formatted strings.

    Purpose:
    This function returns yesterday's date as the start date and today's date as the end date,
    both formatted as 'dd-mm-yyyy'.

    Returns:
    tuple: A tuple containing the formatted start date (yesterday) and end date (today).
    """
    # Get today's date
    today = datetime.now()
    
    # Get yesterday's date by subtracting 1 day
    yesterday = today - timedelta(days=1)
    
    # Format the dates in dd-mm-yyyy format
    formatted_yesterday = yesterday.strftime("%d-%m-%Y")
    formatted_today = today.strftime("%d-%m-%Y")
    
    return formatted_yesterday, formatted_today


def get_updated_data_from_api(dates: tuple[str, str]) -> pd.DataFrame:
    """
    Fetch updated data from the API for a given date range.

    Purpose:
    This function accepts a tuple of start and end dates, converts them into datetime objects,
    and retrieves the updated data from the API using the provided date range. 

    Parameters:
    dates (tuple): A tuple containing the start and end dates as strings in 'dd-mm-yyyy' format.

    Returns:
    pd.DataFrame: A DataFrame containing the updated data fetched from the API.
    """
    start_date, end_date = dates
    start_date = datetime.strptime(start_date, "%d-%m-%Y")
    end_date = datetime.strptime(end_date, "%d-%m-%Y")
    
    updated_data = update_and_save_data_all_regions(
        start_date=start_date, 
        end_date=end_date, 
        today_flag=0, 
        local_save=0, 
        dvc_save=0
    )

    json_data = updated_data.to_json(orient='records', lines=False)
    return json_data


def update_data_to_dvc(df_json: dict) -> None:
    """
    Update the given DataFrame to DVC (Data Version Control).

    Purpose:
    This function accepts a DataFrame and uploads it to the DVC storage to version and manage the dataset.

    Parameters:
    df (pd.DataFrame): The DataFrame to be uploaded to DVC.

    Returns:
    None
    """
    df = pd.read_json(df_json)
    upload_to_dvc(df)