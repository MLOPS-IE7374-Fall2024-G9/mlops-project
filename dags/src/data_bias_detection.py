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

    print(metrics_by_group)
    print(overall_metrics)
    print(demographic_parity_difference)
    
    result = {
        'metrics_by_group': metrics_by_group,
        'overall_metrics': overall_metrics,
        'demographic_parity_difference': demographic_parity_difference
    }

    return result



def conditional_mitigation(data: pd.DataFrame, target_col: str, sensitive_col: str, bias_detection_output: dict) -> pd.DataFrame:
    """
    Perform bias mitigation on subgroups identified as biased.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the data.
    - target_col (str): The name of the column to be used as the target variable.
    - sensitive_col (str): The name of the column used as the sensitive feature.
    - bias_detection_output (dict): The output dictionary from the detect_bias function.

    Returns:
    - pd.DataFrame: The mitigated DataFrame.
    """
    biased_groups = bias_detection_output['demographic_parity_difference']
    metrics_by_group = bias_detection_output['metrics_by_group']
    
    print(metrics_by_group)
    print(biased_groups)
    # Identify groups to mitigate based on a threshold (e.g., significant difference in selection rate)
    threshold = 0.5  # Define your threshold for bias
    groups_to_mitigate = metrics_by_group[
        abs(metrics_by_group['Selection Rate'] - bias_detection_output['overall_metrics']['Selection Rate']) > threshold
    ]

    print("Groups to mitigate:", groups_to_mitigate)

    # Apply a chosen mitigation strategy (e.g., reweighting, resampling)
    # Example: Resample the underrepresented group(s)
    for group in groups_to_mitigate.index:
        group_data = data[data[sensitive_col] == group]
        additional_samples = group_data.sample(frac=0.5, replace=True, random_state=42)
        data = pd.concat([data, additional_samples], ignore_index=True)
    
    return data
