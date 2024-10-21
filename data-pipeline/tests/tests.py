import pytest
from unittest import mock
from datetime import datetime, timedelta
from dataset.scripts.dvc_manager import DVCManager
from dataset.scripts.data_downloader import validate_date, get_yesterday_date_range
import pandas as pd


# data downloader tests
# Test for date validation
def test_validate_date_valid():
    # Valid date should return a datetime object
    assert validate_date("01-01-2024") == datetime(2024, 1, 1)

# Test for yesterday's date range
def test_get_yesterday_date_range():
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    start, end = get_yesterday_date_range()
    
    assert start == yesterday.strftime("%d-%m-%Y")
    assert end == today.strftime("%d-%m-%Y")