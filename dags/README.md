
# Airflow Data Preprocessing DAGs

## Directory Structure
The directory structure is organized as follows:

```
dag/
├── src/
│   ├── data_bias_detection.py
│   ├── data_download.py
│   ├── data_drift.py
│   ├── data_preprocess.py
│   ├── data_schema_validation.py
│   ├── model_pipeline.py
├── tests/
├── bias_detection_dag.py
├── data_drift_detection_dag.py
├── model_bias_detection_dag.py
├── model_train_evaluate_dag.py
├── new_data_preprocess_dag.py
├── raw_data_preprocess_dag.py
```

### Explanation
- **src/**: Contains the Python modules used in the DAGs for various tasks, such as downloading, preprocessing, and validating data.
- **tests/**: Holds test files to verify the DAG functionalities and the correctness of each function in `src`.
- **bias_detection_dag.py**: Airflow DAG script for bias detection and mitigation of processed data.
- **data_drift_detection_dag.py**: Airflow DAG script for data drift detection.
- **model_train_evaluate_dag.py**: Airflow DAG script for model bias detection and mitigation.
- **model_train_evaluate_dag.py**: Airflow DAG script for model training and evaluation.
- **new_data_preprocess_dag.py**: Airflow DAG script for downloading and preprocessing new data.
- **raw_data_preprocess_dag.py**: Airflow DAG script, for handling raw api data preprocessing tasks.

---

## `new_data_preprocess_dag.py` - DAG for New Data Download and Preprocessing

The purpose of this DAG is to automate the process of downloading, preprocessing, and validating new data. It includes several steps that perform data cleaning, feature engineering, schema validation, and updating the data repository with version control. The DAG leverages Airflow to orchestrate these tasks and uses DVC (Data Version Control) to ensure data reproducibility.

### DAG Steps and Explanation

1. **Initialize Date Range**:
   - `last_k_start_end_date_task`: Calculates the start and end dates for the last `delta_days` (default 7 days), which is used for data extraction.

2. **Fetch New Data**:
   - `updated_data_from_api_task`: Pulls the latest data based on the calculated date range. Data is fetched as JSON and passed to the next step.

3. **Data Cleaning**:
   - `clean_data_task`: Cleans the fetched data by removing missing values and duplicates.

4. **Feature Engineering**:
   - `engineer_features_task`: Adds rolling means, standard deviations, and lagged values for features such as temperature, wind speed, and humidity.

5. **Cyclic Features**:
   - `add_cyclic_features_task`: Adds cyclic (sin/cos) transformations for the month feature, helping to capture seasonal patterns.

6. **Normalization and Encoding**:
   - `normalize_and_encode_task`: Normalizes numerical data and encodes categorical features to prepare for model training.

7. **Feature Selection**:
   - `select_final_features_task`: Retains only the relevant features required for downstream tasks, based on the feature set defined in the task.

8. **Data Validation**:
   - `processed_data_from_dvc_task`: Retrieves preprocessed data from DVC for validation comparison.
   - `validate_data_with_schema_task`: Compares the selected features with a schema and checks for data consistency. Returns a validation flag (`1` for success, `0` for failure).

9. **Branching Based on Validation**:
   - `branch_task`: Checks the validation result and branches the DAG. If validation passes, it proceeds to merge tasks; otherwise, it sends a failure notification.

10. **Merge Data**:
    - `merge_data_task`: Merges newly processed data with existing preprocessed data.
    - `merge_raw_data_task`: Merges new raw data with existing raw data stored in DVC.

11. **Data Deduplication**:
    - `redundant_removal_task`: Removes duplicate records to ensure data consistency.

12. **Data Update to DVC**:
    - `update_data_to_dvc_task` and `update_raw_data_to_dvc_task`: Push the updated preprocessed and raw data to DVC for version control and tracking.

13. **Cleanup**:
    - `delete_local_task`: Deletes temporary files and cleans up the local environment. This task is triggered after all preceding tasks have completed, regardless of success or failure.

14. **Email Notifications**:
    - `send_email`: Sends a success notification when the DAG completes successfully.
    - `send_data_validation_failure_email`: Sends a failure notification if data validation fails, prompting the data team to review the data.


### Data Schema Generation and Data Validation
To ensure the consistency and reliability of the dataset, this DAG integrates a data validation step using the DataSchemaAndStatistics class. The schema defines the expected structure of the data, including column names, data types, and constraints. This schema is either inferred from an initial dataset or loaded from a predefined schema file, ensuring that new data adheres to the established format (Pandera library is used)

The DataSchemaAndStatistics class (dataset/scripts/data_schema.py) uses pandera for schema management and validation. The infer_schema() method automatically infers the schema from an initial dataset, capturing essential characteristics like data types and column structure. Once inferred, this schema is stored in JSON format, making it easy to reload and validate against future datasets. The validate_data() method checks new data against the inferred schema, logging any discrepancies. A successful validation returns a 1, while failures return a 0 and log the validation errors. This validation process ensures that only clean and consistent data is processed and saved, enhancing the reliability of the data pipeline.

---

## `raw_data_preprocess_dag.py` - DAG for Raw Data Preprocessing

This DAG handles the preprocessing of raw data obtained from DVC. It focuses on executing preprocessing steps to ensure data consistency, applies transformations, and then updates the preprocessed data back to DVC for tracking. Raw data is here all the data, which is downloaded in raw from API (without data preprocessing) and stored in DVC.

### DAG Steps and Explanation

1. **Retrieve Raw Data**:
   - `data_from_dvc_task`: Retrieves raw data from DVC storage, preparing it for the preprocessing pipeline.

2. **Preprocess Data**:
   - `preprocess_pipeline_task`: Applies all necessary preprocessing steps on the raw data, including cleaning, transformation, and formatting to prepare it for downstream tasks.

3. **Update Data to DVC**:
   - `update_data_to_dvc_task`: Pushes the preprocessed data back to DVC, ensuring that it is version-controlled and accessible for further analysis.

4. **Cleanup**:
   - `delete_local_task`: Cleans up any temporary files from local storage after all tasks are complete.

5. **Email Notifications**:
   - `send_email`: Sends a notification email upon successful completion of the DAG.
   - `email_notify_failure`: Sends an alert if any task in the DAG fails.

---

## `bias_detection_and_mitigation.py` - DAG for Bias Detection and Mitigation

The purpose of this DAG is to automate the detection and mitigation of bias in the dataset. It performs several steps that include loading data, identifying biases, applying conditional mitigation strategies, and updating the data repository with the mitigated results. The DAG uses Airflow for orchestration and ensures continuous monitoring of fairness in the dataset.

### DAG Steps and Explanation

1. **Load Data**:
   - `processed_data_from_dvc_task`: Retrieves the preprocessed data from DVC for bias detection.

2. **Identify Bias**:
   - `identify_bias_task`: Loads the new data and detects biases based on the specified target and sensitive columns. The results are saved for use in the next step.

3. **Mitigate Bias**:
   - `mitigate_bias_task`: Loads the original data and the identified biases, then applies conditional mitigation techniques based on the results from the bias detection step.

4. **Update Mitigated Data**:
   - `mitigated_data_to_dvc_task`: Saves the mitigated dataset back to DVC for version control, ensuring that the data reflects the applied bias mitigation strategies.

5. **Task Dependencies**:
   - The DAG sets up a sequence of task dependencies that dictate the flow of data processing from loading to mitigation, ensuring that each step is executed in the correct order.

---

## `data_drift_detection_dag.py` - DAG for Data Drift Detection

The purpose of this DAG is to automate the process of detecting data drift in the dataset. It includes tasks for loading the data, performing various statistical tests to identify drift, and notifying stakeholders about the results. The DAG leverages Airflow to orchestrate these tasks and utilizes different methods to assess changes in data distribution.

### DAG Steps and Explanation

1. **Load Data**:
   - `load_data_task`: Loads the preprocessed data from the specified file for drift detection.

2. **Detect Drift**:
   - `detect_drift_evidently`: Utilizes the Evidently library to generate a comprehensive drift report in HTML format, which summarizes the findings related to drift in the dataset.
   - `detect_drift_ks_test`: Applies the Kolmogorov-Smirnov test to identify significant changes between distributions of features in the dataset.
   - `detect_drift_psi`: Uses the Population Stability Index (PSI) method to assess the stability of the dataset over time.

3. **Email Notifications**:
   - `send_email`: Sends notifications regarding the success or failure of the DAG execution, alerting stakeholders to the status of the drift detection process.

4. **Task Dependencies**:
   - The DAG establishes dependencies that ensure the data loading task is completed before drift detection tasks are executed, followed by the email notification.

---

