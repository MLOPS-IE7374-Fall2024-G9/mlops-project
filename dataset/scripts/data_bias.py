import pandas as pd
import numpy as np
import logging
from typing import List, Dict
import argparse

class DataBiasDetection:
    def __init__(self, data: pd.DataFrame):
        """Initialize with dataset."""
        self.data = data
        self.bias_report = {}

    def data_slicing(self, slice_cols: List[str]) -> Dict[str, pd.DataFrame]:
        """Slice data based on unique values in specified columns."""
        sliced_data = {}
        for col in slice_cols:
            unique_vals = self.data[col].unique()
            for val in unique_vals:
                slice_name = f"{col}_{val}"
                sliced_data[slice_name] = self.data[self.data[col] == val]
                logging.info(f"Data slice created: {slice_name} with {len(sliced_data[slice_name])} rows.")
        return sliced_data

    def calculate_statistics(self, sliced_data: Dict[str, pd.DataFrame], metric_col: str) -> Dict[str, float]:
        """Calculate mean statistics for each data slice."""
        slice_statistics = {}
        for slice_name, df_slice in sliced_data.items():
            mean_value = df_slice[metric_col].mean()
            slice_statistics[slice_name] = mean_value
            logging.info(f"Mean {metric_col} for slice {slice_name}: {mean_value:.2f}")
        return slice_statistics

    def detect_bias(self, slice_statistics: Dict[str, float], threshold_ratio: float = 0.2) -> List[str]:
        """Detect bias by identifying slices with significant mean deviation."""
        overall_mean = np.mean(list(slice_statistics.values()))
        biased_slices = [
            slice_name for slice_name, mean_value in slice_statistics.items()
            if abs(mean_value - overall_mean) > threshold_ratio * overall_mean
        ]
        
        # Log bias detection
        if biased_slices:
            logging.warning(f"Bias detected in slices: {biased_slices}")
        else:
            logging.info("No significant bias detected.")
        
        self.bias_report['biased_slices'] = biased_slices
        return biased_slices

    def document_bias_report(self) -> None:
        """Log and document bias detection results."""
        logging.info("Bias Report:")
        for key, value in self.bias_report.items():
            logging.info(f"{key}: {value}")


if __name__ == "__main__":
    # Set up argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Detect data bias in specified slices.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV data file.")
    parser.add_argument("--slice_cols", type=str, nargs='+', required=True, help="List of columns to slice data on.")
    parser.add_argument("--metric_col", type=str, required=True, help="Column to calculate metrics on.")
    parser.add_argument("--threshold_ratio", type=float, default=0.2, help="Threshold ratio for detecting bias.")
    
    args = parser.parse_args()

    # Load data
    try:
        data = pd.read_csv(args.data_path)
        logging.info("Data loaded successfully from %s.", args.data_path)
    except Exception as e:
        logging.error("Failed to load data from %s. Error: %s", args.data_path, e)
        raise

    # Initialize DataBiasDetection
    detector = DataBiasDetection(data=data)

    # Slice data and calculate statistics
    sliced_data = detector.data_slicing(args.slice_cols)
    slice_statistics = detector.calculate_statistics(sliced_data, args.metric_col)

    # Detect bias and document results
    biased_slices = detector.detect_bias(slice_statistics, args.threshold_ratio)
    detector.document_bias_report()


# usage - python data_bias.py --data_path path/to/data.csv --slice_cols column1 column2 --metric_col target_column --threshold_ratio 0.2
