from datetime import datetime, timedelta
from dags.src.data_pipeline import *

def test_get_start_end_dates():
    # Get today's and yesterday's date
    today = datetime.now()
    yesterday = today - timedelta(days=1)

    # Format them as 'dd-mm-yyyy'
    expected_start_date = yesterday.strftime("%d-%m-%Y")
    expected_end_date = today.strftime("%d-%m-%Y")

    # Call the function
    start_date, end_date = get_start_end_dates()
    
    # Assert that the function returns the correct values
    assert start_date == expected_start_date
    assert end_date == expected_end_date

def test_get_updated_data_from_api():
    # Calculate the static start and end dates (10 days ago and 9 days ago)
    today = datetime.now()
    start_date = (today - timedelta(days=10)).strftime("%d-%m-%Y")
    end_date = (today - timedelta(days=9)).strftime("%d-%m-%Y")

    # Call the function with the static dates
    result_df = get_updated_data_from_api((start_date, end_date))
    
    # Convert 'datetime' column to the correct format
    # Assuming the format is like '2019-06-05T17', we parse it to complete 'YYYY-MM-DD HH:00:00'
    result_df['datetime'] = pd.to_datetime(result_df['datetime'], format='%Y-%m-%dT%H', errors='coerce')

    # Filter the result to check only valid dates
    result_dates = result_df['datetime'].dt.strftime("%Y-%m-%d").tolist()

    # Expected dates in 'YYYY-MM-DD' format
    expected_start_date = (today - timedelta(days=10)).strftime("%Y-%m-%d")
    expected_end_date = (today - timedelta(days=9)).strftime("%Y-%m-%d")

    # Assert that the DataFrame contains these specific dates
    assert expected_start_date in result_dates, f"{expected_start_date} not found in result dates"
    assert expected_end_date in result_dates, f"{expected_end_date} not found in result dates"