import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def detect_bias(data: pd.DataFrame, target_col: str, sensitive_col: str) -> dict:
    """
    Perform bias detection on a DataFrame using a specified sensitive feature.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the data.
    - target_col (str): The name of the column to be used as the target variable.
    - sensitive_col (str): The name of the column used as the sensitive feature.

    Returns:
    - dict: A dictionary containing metrics by group, overall metrics, and bias differences.
    """
    from fairlearn.metrics import MetricFrame, selection_rate
    from sklearn.metrics import accuracy_score

    # Convert the target variable to binary outcomes for bias detection
    y_true = data[target_col].notnull().astype(int)
    y_pred = data[target_col].notnull().astype(int)
    
    # Create a MetricFrame for bias detection
    metric_frame = MetricFrame(
        metrics={
            'Selection Rate': selection_rate,
            'Accuracy': accuracy_score
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=data[sensitive_col]
    )
    
    metrics_by_group = metric_frame.by_group
    overall_metrics = metric_frame.overall
    demographic_parity_difference = metric_frame.difference(method='between_groups')

    print('By Group Metric Analysis')
    print(metrics_by_group)
    print('--------------------------------\n')
    print('Overall Metric Analysis')
    print('\n', overall_metrics)
    print('--------------------------------\n')
    print('Demographic Parity Difference between groups')
    print('\n', demographic_parity_difference)
    
    result = {
        'metrics_by_group': metrics_by_group,
        'overall_metrics': overall_metrics,
        'demographic_parity_difference': demographic_parity_difference
    }

    return result



def conditional_mitigation_with_resampling(data: pd.DataFrame, date_col: str, target_col: str, sensitive_col: str, bias_detection_output: dict, freq: str = 'H') -> pd.DataFrame:
    """
    Perform bias mitigation on subgroups identified as biased through resampling and imputation.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the data.
    - date_col (str): The name of the date column.
    - target_col (str): The name of the column to be used as the target variable.
    - sensitive_col (str): The name of the column used as the sensitive feature.
    - bias_detection_output (dict): The output dictionary from the detect_bias function.
    - freq (str): The resampling frequency (e.g., 'H' for hourly).

    Returns:
    - pd.DataFrame: The mitigated DataFrame.
    """
    metrics_by_group = bias_detection_output['metrics_by_group']
    overall_metrics = bias_detection_output['overall_metrics']
    
    # Threshold for bias detection
    threshold = 0.05
    groups_to_mitigate = metrics_by_group[
        abs(metrics_by_group['Selection Rate'] - overall_metrics['Selection Rate']) > threshold
    ]

    print("Groups to mitigate with resampling and imputation:", groups_to_mitigate.index.tolist())

    # Determine the overall date range
    min_date, max_date = data[date_col].min(), data[date_col].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq=freq)
    
    # Create a new DataFrame for storing mitigated data
    mitigated_data = []

    # Apply mitigation by resampling and imputing only on biased groups
    for group in groups_to_mitigate.index:
        group_data = data[data[sensitive_col] == group]
        
        # Set the date column as index for resampling
        group_data = group_data.set_index(date_col).reindex(all_dates)
        group_data.index.name = date_col
        group_data[sensitive_col] = group  # Add the `subba-name` column back

        # Impute missing values with forward and backward fill
        group_data = group_data.fillna(method='ffill').fillna(method='bfill')

        # Add the mitigated group data to the list
        mitigated_data.append(group_data.reset_index())

    # For groups that are not biased, add them as-is
    unbiased_data = data[~data[sensitive_col].isin(groups_to_mitigate.index)]
    mitigated_data.append(unbiased_data)

    # Concatenate all data
    balanced_data = pd.concat(mitigated_data, ignore_index=True)

    return balanced_data

if __name__ == '__main__':
    data = pd.read_csv('/Users/akm/Desktop/mlops-project/data_preprocess.csv')
    data.dropna(inplace=True)
    # Step 1: Detect bias in the dataset
    bias_detection_output = detect_bias(data, target_col='value', sensitive_col='subba-name')
    
    # Step 2: Apply conditional mitigation with resampling and imputation for biased groups
    mitigated_data = conditional_mitigation_with_resampling(data, date_col='datetime', target_col='value', sensitive_col='subba-name', bias_detection_output=bias_detection_output, freq='H')
    
    # Final dataset ready for model training
    print("After mitigation with resampling and imputation:")
    print(mitigated_data.info())
