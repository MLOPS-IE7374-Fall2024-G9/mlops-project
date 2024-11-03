import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from data_download import *
 
class DataDriftDetector:
    def __init__(self, baseline_data: pd.DataFrame, new_data: pd.DataFrame):
        """
        Initialize the DataDriftDetector with a baseline dataset and a new dataset for comparison.
        """
        self.baseline_data = baseline_data
        self.new_data = new_data
        self.report = None
 
 #Evidently Check
    def detect_drift_evidently(self):
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
        #self.report.save_html("/your-path/drift_report.html") #To save locally
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
 

file_path = get_data_from_dvc('data_raw.csv') #pulling data from DVC

df = pd.read_csv(file_path)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Convert the 'datetime' column to datetime format
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

#using the existing file to detect drift - need to connect with the daily data 
split_point = int(len(df) * 0.7)
baseline_df = df.iloc[:split_point]
new_data_df = df.iloc[split_point:]
 
# Initialize the detector
drift_detector = DataDriftDetector(baseline_df, new_data_df)
 
# Detect drift using Evidently
drift_detector.detect_drift_evidently()
evidently_results = drift_detector.get_results()
print("Evidently Drift Detection Results:",evidently_results)
 
# Detect drift using KS Test
ks_test_results = drift_detector.detect_drift_ks_test()
ks_df = pd.DataFrame.from_dict(ks_test_results, orient="index")
print("KS Test Drift Results:",ks_df)
 
# Detect drift using PSI
psi_results = drift_detector.detect_drift_psi()
psi_df = pd.DataFrame.from_dict(psi_results, orient="index")
print("PSI Drift Results:", psi_df)