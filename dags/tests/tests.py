import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pandas as pd
from datetime import datetime, timedelta
from src.data_pipeline import ( 
    get_start_end_dates,
    get_last_k_start_end_dates,
    get_updated_data_from_api,
    merge_data,
    redundant_removal,
    get_data_from_dvc,
)

def test_get_start_end_dates():
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    func_yesterday, func_today = get_start_end_dates()
    assert func_yesterday == yesterday, f"Expected {yesterday} but got {func_yesterday}"
    assert func_today == today, f"Expected {today} but got {func_today}"

def test_get_last_k_start_end_dates():
    days = 5
    today = datetime.now().strftime('%d-%m-%Y')
    start_date = (datetime.now() - timedelta(days=days-1)).strftime('%d-%m-%Y')
    func_start_date, func_today = get_last_k_start_end_dates(days)
    assert func_start_date == start_date, f"Expected {start_date} but got {func_start_date}"
    assert func_today == today, f"Expected {today} but got {func_today}"

def test_get_updated_data_from_api():
    today = datetime.now().strftime('%d-%m-%Y')
    dates = (today, today)
    api_json = get_updated_data_from_api(dates)
    api_df = pd.read_json(api_json, orient='records')
    assert any(today in str(dt) for dt in api_df['datetime'].values), f"Expected today's date ({today}) in API data but it was not found."

def test_merge_data():
    data1 = {"datetime": ["2024-10-01", "2024-10-02"], "value": [100, 200]}
    data2 = {"datetime": ["2024-10-03", "2024-10-04"], "value": [300, 400]}
    api_json = pd.DataFrame(data1).to_json(orient='records', lines=False)
    dvc_json = pd.DataFrame(data2).to_json(orient='records', lines=False)
    merged_json = merge_data(api_json, dvc_json)
    merged_df = pd.read_json(merged_json, orient='records')
    expected_data = {"datetime": ["2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04"], "value": [100, 200, 300, 400]}
    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(merged_df, expected_df)

def test_redundant_removal():
    data_with_duplicates = {"datetime": ["2024-10-01", "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-03"], "value": [100, 100, 200, 300, 300]}
    data_df = pd.DataFrame(data_with_duplicates)
    data_json = data_df.to_json(orient='records', lines=False)
    unique_json = redundant_removal(data_json)
    unique_df = pd.read_json(unique_json, orient='records')
    expected_data = {"datetime": ["2024-10-01", "2024-10-02", "2024-10-03"], "value": [100, 200, 300]}
    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(unique_df, expected_df)

def test_get_data_from_dvc():
    filename = "data_preprocess.csv"  # Adjust filename if needed
    data_json = get_data_from_dvc(filename)
    assert data_json is not None and data_json != "", "The JSON output is null or empty."
    data_df = pd.read_json(data_json, orient='records')
    assert not data_df.empty, "The DataFrame is empty."
    data_df['datetime'] = pd.to_datetime(data_df['datetime'], errors='coerce')
    assert data_df['datetime'].min() >= pd.Timestamp("2019-01-01"), "Data does not start from 2019 as expected."

def main():
    test_get_start_end_dates()
    print("test_get_start_end_dates passed")
    
    test_get_last_k_start_end_dates()
    print("test_get_last_k_start_end_dates passed")
    
    test_get_updated_data_from_api()
    print("test_get_updated_data_from_api passed")
    
    test_merge_data()
    print("test_merge_data passed")
    
    test_redundant_removal()
    print("test_redundant_removal passed")
    
    test_get_data_from_dvc()
    print("test_get_data_from_dvc passed")

if __name__ == '__main__':
    main()