import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from ..scripts.data_downloader import main, validate_date, get_yesterday_date_range

# data downloader tests
# Test for date validation
def test_validate_date_valid():
    # Valid date should return a datetime object
    assert validate_date("01-01-2024") == datetime(2024, 1, 1)

def test_validate_date_invalid():
    # Invalid date should raise an argparse.ArgumentTypeError
    with pytest.raises(SystemExit):  # argparse.ArgumentTypeError triggers SystemExit
        validate_date("2024-01-01")  # Wrong format

# Test for yesterday's date range
def test_get_yesterday_date_range():
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    start, end = get_yesterday_date_range()
    
    assert start == yesterday.strftime("%d-%m-%Y")
    assert end == today.strftime("%d-%m-%Y")