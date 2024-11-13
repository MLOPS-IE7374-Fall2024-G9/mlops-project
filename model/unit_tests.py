import pytest
import pandas as pd
import json
import os
import sys
from unittest.mock import patch, mock_open, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './')))

from scripts.dataset_loader import load_config, load_and_split_dataset

# Sample paths for testing
SAMPLE_DATA_PATH = "sample_dataset.csv"
SAMPLE_CONFIG_PATH = "sample_config.json"
SAMPLE_DATA_DIR = "./data"

# Sample data and configuration
SAMPLE_DATA = pd.DataFrame({
    "feature1": range(100),
    "feature2": range(100, 200),
    "label": [0, 1] * 50
})
SAMPLE_CONFIG = {
    "test_size": 0.2,
    "validation_size": 0.25
}

@pytest.fixture
def mock_csv_file():
    """Mock for reading a CSV file."""
    with patch("pandas.read_csv", return_value=SAMPLE_DATA) as mock_read_csv:
        yield mock_read_csv

@pytest.fixture
def mock_json_file():
    """Mock for reading a JSON config file."""
    with patch("builtins.open", mock_open(read_data=json.dumps(SAMPLE_CONFIG))):
        yield

@pytest.fixture
def mock_os_path_exists():
    """Mock for os.path.exists."""
    with patch("os.path.exists", return_value=True) as mock_exists:
        yield mock_exists

@pytest.fixture
def mock_os_remove():
    """Mock for os.remove to avoid deleting real files."""
    with patch("os.remove") as mock_remove:
        yield mock_remove

@pytest.fixture
def mock_to_csv():
    """Mock for to_csv to avoid actual file creation."""
    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        yield mock_to_csv

def test_load_config(mock_json_file, mock_os_path_exists):
    """Test loading configuration from a JSON file."""
    test_size, validation_size = load_config(SAMPLE_CONFIG_PATH)
    assert test_size == SAMPLE_CONFIG["test_size"]
    assert validation_size == SAMPLE_CONFIG["validation_size"]

def test_load_and_split_dataset(mock_csv_file, mock_os_path_exists):
    """Test loading and splitting dataset without saving."""
    test_size = SAMPLE_CONFIG["test_size"]
    validation_size = SAMPLE_CONFIG["validation_size"]
    train_data, validation_data, test_data = load_and_split_dataset(SAMPLE_DATA_PATH, test_size, validation_size)
    
    # Check if splits are approximately the expected sizes
    assert len(test_data) == int(test_size * len(SAMPLE_DATA))
    assert len(train_data) == len(SAMPLE_DATA) - len(test_data) - len(validation_data)

def test_missing_config_file():
    """Test handling of missing configuration file."""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            load_config(SAMPLE_CONFIG_PATH)

def test_missing_data_file():
    """Test handling of missing data file."""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            load_and_split_dataset(SAMPLE_DATA_PATH, 0.2, 0.25)
