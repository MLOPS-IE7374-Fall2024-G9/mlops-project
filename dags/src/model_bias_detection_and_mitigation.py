import os
import datetime
import joblib
import pickle
import pandas as pd
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fairlearn.metrics import MetricFrame
from google.cloud import storage
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("airflow_tasks.log")
    ]
)
logger = logging.getLogger(__name__)

BIAS_ANALYSIS_GOOGLE_CRED_PATH = os.path.join(os.path.dirname(__file__),'../../mlops-7374-3e7424e80d76.json')

def setup_local_run_folder(base_path, run_type):
    """
    Creates a local folder for storing files related to the current run and logs the details.

    Args:
        base_path (str): Base path where the folder should be created.
        run_type (str): Type of the run (e.g., "detection" or "mitigation").

    Returns:
        tuple: A tuple containing the local folder path and the corresponding GCP folder path.
    """
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    local_run_folder = os.path.join(base_path, f"{run_type}_{run_id}")
    os.makedirs(local_run_folder, exist_ok=True)
    logger.info(f"Created local run folder at {local_run_folder}")
    return local_run_folder, f"runs/{run_type}_{run_id}/"

def load_splits(X_train_path, X_test_path, y_train_path, y_test_path, sensitive_train_path, sensitive_test_path):
    """
    Loads training, testing, and sensitive feature data from the specified file paths.

    Args:
        X_train_path (str): Path to the training features file.
        X_test_path (str): Path to the test features file.
        y_train_path (str): Path to the training target file.
        y_test_path (str): Path to the test target file.
        sensitive_train_path (str): Path to the training sensitive features file.
        sensitive_test_path (str): Path to the test sensitive features file.

    Returns:
        tuple: Loaded data for X_train, X_test, y_train, y_test, sensitive_train, and sensitive_test.
    """
    logger.info("Loading train, test, and sensitive data splits.")
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)
    sensitive_train = pd.read_csv(sensitive_train_path)
    sensitive_test = pd.read_csv(sensitive_test_path)

    X_train.drop(columns=['Unnamed: 0'], inplace=True)
    X_test.drop(columns=['Unnamed: 0'], inplace=True)
    y_train.drop(columns=['Unnamed: 0'], inplace=True)
    y_test.drop(columns=['Unnamed: 0'], inplace=True)
    sensitive_train.drop(columns=['Unnamed: 0'], inplace=True)
    sensitive_test.drop(columns=['Unnamed: 0'], inplace=True)

    return X_train, X_test, y_train, y_test, sensitive_train, sensitive_test


def upload_to_gcp(local_folder, gcp_folder, bucket_name):
    """
    Uploads all files from a local folder to a GCP bucket folder.

    Args:
        local_folder (str): Path to the local folder containing files to upload.
        gcp_folder (str): Path in the GCP bucket where files should be stored.
        bucket_name (str): Name of the GCP bucket.

    Returns:
        None
    """

    original_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = BIAS_ANALYSIS_GOOGLE_CRED_PATH

    try:
        # Initialize the GCP storage client
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)

        # Upload files while preserving the folder structure
        for root, _, files in os.walk(local_folder):
            for file in files:
                local_file_path = os.path.join(root, file)

                # Construct the relative path to preserve folder structure
                relative_path = os.path.relpath(local_file_path, local_folder)
                blob_path = os.path.join(gcp_folder, os.path.basename(local_folder), relative_path)

                # Upload the file to GCP
                bucket.blob(blob_path).upload_from_filename(local_file_path)
                logger.info(f"Uploaded {local_file_path} to GCP at {blob_path}")

    except Exception as e:
        logger.error(f"Failed to upload files to GCP: {e}")
        raise
    finally:
        # Clear the GCP credentials from the environment
        if original_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = original_credentials
        else:
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

def generate_metric_frame(model, X_test, y_test, sensitive_features):
    """
    Generates a MetricFrame for model evaluation, capturing group fairness metrics.

    Args:
        model (sklearn-like object): Trained model to evaluate.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test set.
        sensitive_features (pd.Series): Sensitive attribute(s) for fairness evaluation.

    Returns:
        MetricFrame: MetricFrame object containing group-wise metrics.
    """
    logger.info("Generating metric frame for model predictions.")
    y_pred = model.predict(X_test)
    metric_frame = MetricFrame(
        metrics={'mse': mean_squared_error, 'mae': mean_absolute_error},
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    return metric_frame

def save_metric_frame(metric_frame, model_name, bucket_name, local_path):
    """
    Saves a MetricFrame object as a .pkl file locally and uploads it to GCP.

    Args:
        metric_frame (MetricFrame): MetricFrame object to save.
        model_name (str): Name of the model to include in the file name.
        bucket_name (str): Name of the GCP bucket.
        local_path (str): Local directory path to save the file.

    Returns:
        str: GCP path where the file was stored.
    """
    safe_model_name = "".join(char if char.isalnum() else "_" for char in model_name)
    filename = f"{safe_model_name}_metric_frame_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    file_path = os.path.join(local_path, filename)
    
    with open(file_path, 'wb') as f:
        pickle.dump(metric_frame, f)
    # upload_to_gcp(local_path, f"metric_analysis/{filename}", bucket_name)
    # os.remove(file_path)
    # return f"gs://{bucket_name}/metric_analysis/{filename}"

    return file_path

def save_model(model, model_name, local_path):
    """
    Saves a trained model as a .pkl file at a specified local path.

    Args:
        model (sklearn-like object): Trained model to save.
        model_name (str): Name of the model to include in the file name.
        local_path (str): Local directory path to save the file.

    Returns:
        str: Path where the model file was stored locally.
    """
    # Ensure the model name is safe for filenames
    safe_model_name = "".join(char if char.isalnum() else "_" for char in model_name)
    
    # Create a timestamped filename for the model
    filename = f"{safe_model_name}_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    file_path = os.path.join(local_path, filename)
    
    # Save the model to the specified path
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {file_path}")
    return file_path


def generate_and_save_bias_analysis(metric_frame, metric_key, local_path):
    """
    Generates a bar chart for the given metric and saves it locally.

    Args:
        metric_frame (MetricFrame): MetricFrame object for analysis.
        metric_key (str): The metric to visualize (e.g., 'mae').
        local_path (str): Local directory to save the graph.

    Returns:
        str: Path to the saved graph file.
    """
    logger.info("Generating bias analysis graph.")
    metric_frame.by_group[metric_key].plot(kind='bar')
    graph_path = os.path.join(local_path, f"{metric_key}_by_group.png")
    plt.title(f"{metric_key.upper()} by Group")
    plt.xlabel("Groups")
    plt.ylabel(metric_key.upper())
    plt.savefig(graph_path)
    logger.info(f"Graph saved at {graph_path}")
    return graph_path

def identify_bias_by_threshold(metric_frame, metric_key, threshold_multiplier=1.5):
    """
    Identifies biased groups based on a threshold applied to a given metric.

    Args:
        metric_frame (MetricFrame): MetricFrame object containing metrics by group.
        metric_key (str): Key of the metric to evaluate (e.g., 'mae', 'mse').
        threshold_multiplier (float): Multiplier to determine the bias threshold.

    Returns:
        tuple: 
            - biased_flag (bool): True if any group exceeds the threshold.
            - biased_groups (pd.DataFrame): DataFrame of groups exceeding the threshold.
            - threshold (float): Calculated threshold value.
            - metric_ratio (float): Ratio of the max to min metric value across groups.
    """
    logger.info("Identifying bias in groups based on thresholding.")
    
    # Calculate the overall metric value
    overall_metric = metric_frame.overall[metric_key]
    logger.info("Overall {metric_key} : {overall_metric}")
    threshold = threshold_multiplier * overall_metric

    # Compute metric differences by group
    metric_frame_results = metric_frame.by_group.copy()
    metric_frame_results[f'{metric_key}_difference'] = abs(
        metric_frame_results[metric_key] - overall_metric
    )

    # Identify groups exceeding the threshold
    biased_groups = metric_frame_results[metric_frame_results[f'{metric_key}_difference'] > threshold]
    metric_ratio = metric_frame.ratio()

    biased_flag = not biased_groups.empty

    # Log results
    logger.info("Bias identification complete.")
    logger.info(f" Metric ratio: {metric_ratio}")

    return biased_flag, biased_groups, metric_ratio, overall_metric


def run_detection(X_test, y_test, sensitive_test, model, bucket_name, temp_path):
    """
    Performs bias detection, evaluates metrics, and uploads results to GCP.

    Args:
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test set.
        sensitive_test (pd.Series): Sensitive attribute(s) for fairness evaluation.
        model (sklearn-like object): Trained model for evaluation.
        bucket_name (str): Name of the GCP bucket.
        temp_path (str): Local directory for temporary files.

    Returns:
        None
    """
    local_run_folder, gcp_run_folder = setup_local_run_folder(temp_path, "detection")

    metric_frame = generate_metric_frame(model, X_test, y_test, sensitive_test)
    biased_flag, biased_groups, mae_ratio, overall_mae= identify_bias_by_threshold(metric_frame, 'mae')

    biased_groups_list = biased_groups.index.tolist() if biased_flag else []
    if biased_flag:
        logger.warning(f"Bias Detected")
        logger.info(f"Biased groups identified: {biased_groups}")
    
    metric_frame_gcp_path = save_metric_frame(metric_frame, "XGBRegressor_before_mitigation", bucket_name, local_run_folder)
    graph_saved_path = generate_and_save_bias_analysis(metric_frame, 'mae', local_run_folder)

    findings = {
        "bias_detected": biased_flag,
        "affected_groups": biased_groups_list,
        "metric_ratio": mae_ratio.to_dict() if hasattr(mae_ratio, "to_dict") else mae_ratio,  # Convert Series to dictionary
        "overall_mae": overall_mae,
        "metric_frame_path": os.path.relpath(metric_frame_gcp_path, start=local_run_folder),
        "graph_path": os.path.relpath(graph_saved_path, start=local_run_folder),
    }


    # Save JSON findings
    findings_path = os.path.join(local_run_folder, "detection_report.json")
    with open(findings_path, 'w') as json_file:
        json.dump(findings, json_file, indent=4)

    logger.info(f"Detection findings saved to {findings_path}")

    # upload_to_gcp(local_run_folder, gcp_run_folder, bucket_name)
    
    # for file in os.listdir(local_run_folder):
    #     os.remove(os.path.join(local_run_folder, file))
    # os.rmdir(local_run_folder)

    # return metric_frame_gcp_path
    return local_run_folder

def run_mitigation(X_train, X_test, y_train, y_test, sensitive_train, sensitive_test, model, bucket_name, temp_path):
    
    """
    Mitigates bias by reweighting training samples and evaluates the model.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training target labels.
        y_test (pd.Series): Test target labels.
        sensitive_train (pd.Series): Sensitive attribute(s) for training data.
        sensitive_test (pd.Series): Sensitive attribute(s) for testing data.
        model (sklearn-like object): Trained model for evaluation.
        bucket_name (str): Name of the GCP bucket.
        temp_path (str): Local directory for temporary files.

    Returns:
        str: Path to the local folder where results are stored.
    """
    local_run_folder, gcp_run_folder = setup_local_run_folder(temp_path, "mitigation")

    # Apply sample weighting for bias mitigation
    sample_weights = compute_sample_weight(class_weight="balanced", y=sensitive_train)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Generate MetricFrame for the mitigated model
    metric_frame_weighted = generate_metric_frame(model, X_test, y_test, sensitive_test)

    # Save the MetricFrame and generate bias analysis
    metric_frame_gcp_path = save_metric_frame(metric_frame_weighted, "XGBRegressor_after_mitigation", bucket_name, local_run_folder)
    graph_saved_path = generate_and_save_bias_analysis(metric_frame_weighted, 'mae', local_run_folder)

    # Get overall MAE and by-group metrics
    overall_mae = round(metric_frame_weighted.overall['mae'], 4)  # Bound MAE to 4 decimals
    mae_by_group = metric_frame_weighted.by_group['mae'].round(4).to_dict()  # Group-wise metrics

    # Prepare findings for the mitigation step
    findings = {
        "overall_mae_after_mitigation": overall_mae,
        "mae_by_group": mae_by_group,
        "metric_frame_path": os.path.relpath(metric_frame_gcp_path, start=local_run_folder),
        "graph_path": os.path.relpath(graph_saved_path, start=local_run_folder),
    }

    # Save findings as a JSON file
    findings_path = os.path.join(local_run_folder, "mitigation_report.json")
    with open(findings_path, 'w') as json_file:
        json.dump(findings, json_file, indent=4)

    logger.info(f"Mitigation findings saved to {findings_path}")

    # Upload to GCP (if required)
    # upload_to_gcp(local_run_folder, gcp_run_folder, bucket_name)

    return local_run_folder , model


def load_pkl_files_from_directory(directory_path):
    """
    Loads all .pkl files from the specified directory.

    Args:
        directory_path (str): Path to the directory containing .pkl files.

    Returns:
        dict: A dictionary where keys are file names and values are the loaded objects.
    """
    pkl_files = [f for f in os.listdir(directory_path) if f.endswith('.pkl')]
    loaded_objects = {}
    
    for file in pkl_files:
        file_path = os.path.join(directory_path, file)
        with open(file_path, 'rb') as f:
            loaded_objects[file] = pickle.load(f)
    
    return loaded_objects

if __name__ == '__main__':
    X_train_path = '/Users/akm/Desktop/mlops-project/experiments/X_train_split.csv'
    X_test_path = '/Users/akm/Desktop/mlops-project/experiments/X_test_split.csv'
    y_train_path = '/Users/akm/Desktop/mlops-project/experiments/y_train_split.csv'
    y_test_path = '/Users/akm/Desktop/mlops-project/experiments/y_test_split.csv'
    sensitive_train_path = '/Users/akm/Desktop/mlops-project/experiments/sensitive_train_split.csv'
    sensitive_test_path = '/Users/akm/Desktop/mlops-project/experiments/sensitive_test_split.csv'    

    # loading the train splits 
    # X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = load_splits(X_train_path, X_test_path, y_train_path, y_test_path, sensitive_train_path, sensitive_test_path)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, sensitive_train.shape, sensitive_test.shape)
    # print(X_train.columns, X_test.columns, y_train.columns, y_test.columns, sensitive_train.columns, sensitive_test.columns)

    # lbl_encoder = LabelEncoder()
    # X_train['subba-name'] = lbl_encoder.fit_transform(X_train['subba-name'])
    # X_test['subba-name'] = lbl_encoder.transform(X_test['subba-name'])
    # # # load model
    # # # # model = load_model_from_gcp(bucket_name, model_path)
    # model = XGBRegressor(objective='reg:squarederror', random_state=42,reg_alpha=0.1, reg_lambda=1.0,learning_rate=0.05, n_estimators=1000, min_child_weight=5)
    # model.fit(X_train, y_train)

    # # # y_pred_before_mitigation
    # # y_pred = model.predict(X_test)

    # local_folder = run_detection( X_test, y_test, sensitive_test, model, None, '/Users/akm/Desktop/mlops-project/experiments/temp_bias_analysis/')

    # bias_detection_result_path = '/Users/akm/Desktop/mlops-project/experiments/temp_bias_analysis/detection_20241115_143124'
    
    # # bias_mitigation_result_path = run_mitigation(X_train, X_test, y_train, y_test, sensitive_train, sensitive_test, model, None, '/Users/akm/Desktop/mlops-project/experiments/temp_bias_mitigation_analysis/')

    # bias_mitigation_result_path = '/Users/akm/Desktop/mlops-project/experiments/temp_bias_mitigation_analysis/mitigation_20241115_143719'

    # before_mitigation_metric_frame = load_pkl_files_from_directory(bias_detection_result_path)
    # after_mitigation_metric_frame = load_pkl_files_from_directory(bias_mitigation_result_path)

    # for file_name, metric_frame in before_mitigation_metric_frame.items():
    #     print(f"\nFile: {file_name}")
    #     print("\nOverall Metrics:")
    #     print(metric_frame.overall)  # Displays overall metrics
        
    #     print("\nGroup-wise Metrics:")
    #     print(metric_frame.by_group) 


    # for file_name, metric_frame in after_mitigation_metric_frame.items():
    #     print(f"\nFile: {file_name}")
    #     print("\nOverall Metrics:")
    #     print(metric_frame.overall)  # Displays overall metrics
        
    #     print("\nGroup-wise Metrics:")
    #     print(metric_frame.by_group) 


    # upload_to_gcp('/Users/akm/Desktop/mlops-project/experiments/temp_bias_analysis/detection_20241115_143124','detection_results','model_bias_results')