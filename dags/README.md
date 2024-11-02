
# Airflow Data Preprocessing DAGs

## Directory Structure
The directory structure is organized as follows:

```
dag/
├── src/
│   ├── data_download.py
│   ├── data_preprocess.py
│   ├── data_schema_validation.py
├── tests/
├── new_data_preprocess_dag.py
├── raw_data_preprocess_dag.py
```

### Explanation
- **src/**: Contains the Python modules used in the DAGs for various tasks, such as downloading, preprocessing, and validating data.
- **tests/**: Holds test files to verify the DAG functionalities and the correctness of each function in `src`.
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

