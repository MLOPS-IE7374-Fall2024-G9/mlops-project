import argparse
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
from dataset.scripts.data import *
from dataset.scripts.dvc_manager import *
import os

# helper functions 

def get_yesterday_date_range() -> Tuple[str, str]:
    """
    Returns the date range for yesterday.
    Start as yesterday, end as today.

    Returns:
        Tuple[str, str]: Start date (yesterday) and end date (today) in 'dd-mm-yyyy' format.
    """
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    return yesterday.strftime("%d-%m-%Y"), today.strftime("%d-%m-%Y")


def validate_date(date_text: str) -> datetime:
    """
    Validates the date format and returns a datetime object if valid.

    Args:
        date_text (str): Date in 'dd-mm-yyyy' format.

    Returns:
        datetime: A datetime object representing the validated date.

    Raises:
        argparse.ArgumentTypeError: If the date is not in the correct format.
    """
    try:
        return datetime.strptime(date_text, "%d-%m-%Y")
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Incorrect date format: '{date_text}'. Expected format is dd-mm-yyyy."
        )


def get_start_end_str(
    start_date: datetime, end_date: datetime, today_flag: bool
) -> Tuple[str, str]:
    """
    Returns the start and end date strings based on whether the --today flag is set.

    Args:
        start_date (datetime): Start date for data collection.
        end_date (datetime): End date for data collection.
        today_flag (bool): Flag to indicate whether to use yesterday and today as the date range.

    Returns:
        Tuple[str, str]: The start and end date strings in 'dd-mm-yyyy' format.
    """
    # If 'today_flag' is True, override start_date and end_date with yesterday and today.
    if today_flag:
        start_date_str, end_date_str = get_yesterday_date_range()
    else:
        # Convert dates back to string format after validation
        start_date_str = start_date.strftime("%d-%m-%Y")
        end_date_str = end_date.strftime("%d-%m-%Y")

    return start_date_str, end_date_str

def get_new_data(data_collector_obj, data_regions, start_date, end_date, region, today_flag):
    # zones, object init
    data_zones = data_regions.regions[region]

    # Get the start and end date in strings
    start_date_str, end_date_str = get_start_end_str(start_date, end_date, today_flag)

    # Generate new data
    df_map = data_collector_obj.generate_dataset(
        data_zones, start_date_str, end_date_str
    )

    # Check if data is fetched
    first_df = next(iter(df_map.values()))  # Get the first DataFrame from the map
    if first_df.empty:
        raise Exception("This region has monthly data updates and not daily")

    return df_map

def merge_new_with_existing_data(latest_data_df, df_map):
    # if latest data in dvc exists, combine the dvc + new data
    print("Merging data")
    if latest_data_df is not None:
        df_combined = pd.concat(df_map.values(), ignore_index=True)
        df_combined.sort_values(by="datetime", inplace=True)

        updated_data_df = pd.concat([latest_data_df, df_combined], ignore_index=True)
    else:
        updated_data_df = pd.concat(df_map.values(), ignore_index=True)

    return updated_data_df

def update_and_save_data(
    start_date: datetime,
    end_date: datetime,
    today_flag: bool,
    region: str,
    local_save: bool=False, 
    dvc_save: bool=False):
    """
    function to generate and save the dataset based on the given date range or the --today flag.

    Args:
        start_date (str): Start date for data collection.
        end_date (str): End date for data collection.
        today_flag (bool): Flag to indicate whether to use yesterday and today as the date range.
        region (str): The region for which the data is to be collected. if region == "all"
        local_save (bool): Flag to indicate save in local as csv
    """
    data_regions = DataRegions()
    data_collector_obj = DataCollector()
    dvc_manager_obj = DVCManager("mlops-437516-b9a69694c897.json")

    # Ensure that the region exists in the data regions
    if region not in data_regions.regions:
        raise ValueError(
            f"Region '{region}' is not valid. Available regions: {list(data_regions.regions.keys())}"
        )

    # get latest data from dvc
    latest_data_df = dvc_manager_obj.download_data_from_dvc(save_local=0)
    
    # get new data from api 
    df_map_from_api = get_new_data(data_collector_obj, data_regions, start_date, end_date, region, today_flag)

    # merge api and dvc data
    updated_data_df = merge_new_with_existing_data(latest_data_df, df_map_from_api)
    
    # push updated data back to dvc
    if dvc_save:
        dvc_manager_obj.upload_data_to_dvc(updated_data_df)

    # saving local
    if local_save:
        # Get the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file_path = os.path.join(script_dir, "data.csv")

        # Save it as CSV
        data_collector_obj.save_dataset(updated_data_df, csv_file_path)

    return updated_data_df


def update_and_save_data_all_regions(start_date: datetime,
                                        end_date: datetime,
                                        today_flag: bool,
                                        local_save: bool,
                                        dvc_save: bool):

    data_regions = DataRegions()

    for region in data_regions.regions:
        updated_data_df = update_and_save_data(start_date, end_date, today_flag, region, local_save, dvc_save)
    
    return updated_data_df

def download_from_dvc():
    dvc_manager_obj = DVCManager("mlops-437516-b9a69694c897.json")
    df = dvc_manager_obj.download_data_from_dvc()
    return df

def upload_to_dvc(df):
    dvc_manager_obj = DVCManager("mlops-437516-b9a69694c897.json")
    dvc_manager_obj.upload_data_to_dvc(df)



# -------------------------------------------------------
# if __name__ == "__main__":
#     # Set up argument parsing
#     parser = argparse.ArgumentParser(
#         description="Collect and save dataset for a given date range or for yesterday using the --today flag.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )

#     parser.add_argument(
#         "--start_date",
#         type=validate_date,
#         help="Start date in the format dd-mm-yyyy. This is required unless the --today flag is set.",
#     )

#     parser.add_argument(
#         "--end_date",
#         type=validate_date,
#         help="End date in the format dd-mm-yyyy. This is required unless the --today flag is set.",
#     )

#     parser.add_argument(
#         "--today",
#         action="store_true",
#         help="If provided, fetches the data for yesterday and today, overriding any provided start and end dates.",
#     )

#     parser.add_argument(
#         "--region", type=str, required=True, help="The region for which we need data."
#     )

#     parser.add_argument(
#         "--local_save", action="store_true", help="If provided, saves in csv."
#     )

#     args = parser.parse_args()

#     # Handle the case when 'today' flag is not given but start_date and end_date are missing
#     if args.today:
#         start_date, end_date = None, None  # These will be set in the main function
#     else:
#         if not args.start_date or not args.end_date:
#             parser.error(
#                 "You must provide both --start_date and --end_date unless the --today flag is set."
#             )

#     # Run the main function with parsed arguments
#     update_and_save_data(
#         args.start_date, args.end_date, args.today, args.region, args.local_save
#     )
    
# python data_downloader.py --start_date 05-10-2024 --end_date 06-10-2024 --region new_york
