import argparse
import pandas as pd
import json
from data import *
from typing import Tuple, Dict
from datetime import datetime

def get_updated_data_from_api(dates: Tuple[str, str], regions: Dict[str, Dict[str, list]]) -> pd.DataFrame:
    """
    Fetches updated data from the API for a specified date range and regions.
    """
    data_obj = DataCollector()
    start_date, end_date = dates
    data = data_obj.get_data_from_api(regions, start_date, end_date, today_flag=0)
    return data

def validate_date(date_str: str) -> str:
    """
    Validates the date format as DD-MM-YYYY.
    """
    try:
        # Attempt to parse the date
        datetime.strptime(date_str, "%d-%m-%Y")
        return date_str
    except ValueError:
        raise argparse.ArgumentTypeError("Date must be in DD-MM-YYYY format")

def main():
    """
    Main function to fetch data using date range and regions provided via command-line arguments.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Fetch updated data from API for a specified date range and regions.")
    
    # Define command-line arguments with date format validation
    parser.add_argument(
        '--start_date',
        type=validate_date,
        required=True,
        help="Start date for the data range in the format DD-MM-YYYY."
    )
    parser.add_argument(
        '--end_date',
        type=validate_date,
        required=True,
        help="End date for the data range in the format DD-MM-YYYY."
    )
    parser.add_argument(
        '--regions',
        type=str,
        required=True,
        help="Regions in JSON format, e.g., '{\"texas\": {\"COAS\": [29.749907, -95.358421]}}'"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get dates and regions from arguments
    dates = (args.start_date, args.end_date)
    regions = json.loads(args.regions)  # Parse JSON string to a dictionary
    
    # Fetch data
    data = get_updated_data_from_api(dates, regions)
    
    # Save the data to a CSV file
    data.to_csv("./downloaded_data.csv")
    print("Data successfully saved to 'downloaded_data.csv'")

if __name__ == '__main__':
    main()

# usage (note some zones get monthly demand data from API and not daily)
# python data_downloader.py --start_date DD-MM-YYYY --end_date DD-MM-YYYY --regions '{"region_name": {"location_code": [latitude, longitude]}}'

# example
# python dataset/scripts/data_downloader.py --start_date 09-11-2024 --end_date 10-11-2024 --regions '{"new_york": {"ZONEA": [42.8864, -78.8784]}}'

