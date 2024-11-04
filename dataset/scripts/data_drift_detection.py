import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
 
class DataDriftDetector:
    def __init__(self, baseline_data: pd.DataFrame, new_data: pd.DataFrame):
        """
        Initialize the DataDriftDetector with a baseline dataset and a new dataset for comparison.
        """
        self.baseline_data = baseline_data
        self.new_data = new_data
        self.report = None
 
    #Evidently Check
    def detect_drift_evidently(self, path_to_save):
        """
        Detect drift between baseline and new datasets using Evidently.
        """
        column_mapping = ColumnMapping(
            #datetime='datetime',
            numerical_features=[
        'tempF', 'precipMM', 'humidity', 'visibility', 'pressure', 'HeatIndexF', 
        'DewPointF', 'WindChillF', 'WindGustMiles', 'uvIndex', 'value'
            ],
            #categorical_features=['subba-name', 'zone']
        )
        # Initialize and generate the data drift report
        self.report = Report(metrics=[DataDriftPreset()])
        self.report.run(reference_data=self.baseline_data, current_data=self.new_data, column_mapping=column_mapping)
        self.report.save_html(path_to_save) #To save locally

        report_dict = self.report.as_dict()
        return report_dict


    #Kolmogorov-Smirnov test 
    def detect_drift_ks_test(self, threshold=0.05):
        """
        Detect drift in numerical features using Kolmogorov-Smirnov test.
        :param threshold: Significance level for detecting drift.
        :return: Dictionary of features with p-values and drift flag.
        """
        drift_results = {}
        for col in self.baseline_data.select_dtypes(include=[np.number]).columns:
            stat, p_value = ks_2samp(self.baseline_data[col], self.new_data[col])
            drift_results[col] = {'p_value': p_value, 'drift': p_value < threshold}
        return drift_results
 
    #PSI test
    def detect_drift_psi(self, bins=10, threshold=0.2, epsilon=1e-10):
        """
        Detect drift in numerical features using Population Stability Index (PSI).
        :param bins: Number of bins for binning the feature values.
        :param threshold: PSI threshold to flag drift (common values are 0.1-0.2).
        :return: Dictionary of features with PSI values and drift flag.
        """
        drift_results = {}
        for col in self.baseline_data.select_dtypes(include=[np.number]).columns:
            baseline_counts, _ = np.histogram(self.baseline_data[col], bins=bins)
            new_counts, _ = np.histogram(self.new_data[col], bins=bins)
            baseline_counts = baseline_counts / len(self.baseline_data) + epsilon
            new_counts = new_counts / len(self.new_data) + epsilon
            psi = np.sum((baseline_counts - new_counts) * np.log(baseline_counts / new_counts))
            drift_results[col] = {'PSI': psi, 'drift': psi > threshold}
        return drift_results
 
    def get_results(self):
        """
        Retrieve Evidently drift results in JSON format for further analysis.
        :return: JSON object containing the drift detection report.
        """
        if not self.report:
            raise ValueError("Drift report has not been generated. Call detect_drift_evidently first.")
        return self.report.as_dict()