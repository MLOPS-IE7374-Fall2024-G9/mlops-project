
# Airflow Data Preprocessing DAGs

## Directory Structure
The directory structure is organized as follows:

```
dag/
├── src/
│   ├── data_bias_detection_and_mitigation.py
│   ├── data_download.py
│   ├── data_drift.py
│   ├── data_preprocess.py
│   ├── data_schema_validation.py
│   ├── feature_analyzer.py
│   ├── model_bias_detection_and_mitigation..py
│   ├── model_load_test.py
│   ├── model_pipeline.py
│   ├── model_roll_back.py
│   ├── model_upload_test.py
├── tests/
├── data_bias_detection_dag.py
├── data_drift_detection_dag.py
├── data_feature_imp_analysis_dag.py
├── data_new_preprocess_dag.py
├── data_raw_preprocess_dag.py
├── deployed_monitoring_dag.py
├── deployment_app_dag.py
├── deployment_model_dag.py
├── master_dag.py
├── model_bias_detection_and_mitigation_dag.py
├── model_retrain_evaluate_dag.py
├── model_roll_back_dag.py
├── model_train_evaluate_dag.py
```

### Explanation
- **src/**: Contains the Python modules used in the DAGs for various tasks, such as downloading, preprocessing, and validating data.
- **tests/**: Holds test files to verify the DAG functionalities and the correctness of each function in `src`.
- **data_bias_detection_dag.py**: Airflow DAG script for detecting and mitigating bias in processed data.
- **data_drift_detection_dag.py**: Airflow DAG script for monitoring and detecting data drift in datasets.
- **data_feature_imp_analysis_dag.py**: Airflow DAG script for analyzing and calculating feature importance in data.
- **data_new_preprocess_dag.py**: Airflow DAG script for handling preprocessing of newly obtained data.
- **data_raw_preprocess_dag.py**: Airflow DAG script for preprocessing raw data from APIs or external sources.
- **deployed_monitoring_dag.py**: Airflow DAG script for monitoring the deployment and performance of models in production.
- **deployment_app_dag.py**: Airflow DAG script for automating the deployment of applications and services.
- **deployment_model_dag.py**: Airflow DAG script for managing model deployment processes in production environments.
- **master_dag.py**: Main orchestration DAG that triggers other individual DAGs in the workflow.
- **model_bias_detection_and_mitigation_dag.py**: Airflow DAG script for detecting and mitigating bias in trained models.
- **model_retrain_evaluate_dag.py**: Airflow DAG script for retraining and evaluating machine learning models.
- **model_roll_back_dag.py**: Airflow DAG script for rolling back a model deployment to a previous version if needed.
- **model_train_evaluate_dag.py**: Airflow DAG script for training and evaluating machine learning models.
---

## `data_new_preprocess_dag.py` - DAG for New Data Download and Preprocessing

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

## `data_raw_preprocess_dag.py` - DAG for Raw Data Preprocessing

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

## `data_bias_detection_dag.py` - DAG for Bias Detection and Mitigation

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

## `data_feature_imp_analysis_dag.py` - DAG for Feature Importance Analysis

The purpose of this DAG is to automate the process of analyzing feature importance for a machine learning model. It loads the necessary model and dataset, performs feature importance analysis using SHAP (SHapley Additive exPlanations), and sends email notifications on the success or failure of the DAG execution.

### DAG Steps and Explanation

1. **Feature Importance Analysis**:
   - `analyze_feature_importance`: Analyzes the importance of features in the provided model using SHAP to understand which features are most influential in making predictions.

2. **Email Notifications**:
   - `send_email`: Sends a general notification email for DAG completion.
   - `email_notify_success`: Sends an email notification on task success.
   - `email_notify_failure`: Sends an email notification on task failure.

3. **Task Dependencies**:
   - The DAG establishes dependencies such that the feature importance analysis is performed first, followed by sending the success or failure email based on the result.

---

## `deployed_monitoring_dag.py` - DAG for Monitoring Deployed Models

The purpose of this DAG is to monitor the performance of deployed models by validating their predictions against predefined thresholds. If thresholds are violated, the DAG triggers a retraining process; otherwise, it skips retraining. It also handles tasks such as model downloading, data drift detection, and model bias detection.

### DAG Steps and Explanation

1. **Validation Outputs**:
   - `get_validation_outputs`: Loads processed data and evaluates the model's performance on the test set, returning key metrics like MSE, MAE, and R² score.

2. **Threshold Verification**:
   - `check_thresholds_task`: Compares the validation outputs against predefined thresholds. If the thresholds are violated, retraining or rollback will be triggered.

3. **Model Monitoring**:
   - `download_model_task`: Downloads the latest model artifacts for evaluation.
   - `trigger_data_drift_dag`: Triggers the data drift detection DAG to check if the model's input data has changed significantly.
   - `trigger_data_bias_dag`: Triggers the data bias mitigation DAG to check and mitigate any data bias.
   - `trigger_model_bias_dag`: Triggers the model bias detection and mitigation DAG to evaluate and correct any biases in the model.

4. **Retraining and Rollback**:
   - `trigger_retraining_dag`: Triggers a retraining process if the thresholds are violated.
   - `trigger_rollback_dag`: Rolls back the model to a previous stable version if necessary.
   - `skip_retraining_task`: Skips retraining if the model is performing well within the thresholds.

5. **Task Dependencies**:
   - The DAG establishes a sequence where model validation is done first, followed by threshold checking, and based on the result, it either skips retraining or triggers a rollback and retraining.

---

## `deployment_app_dag.py` - DAG for Deploying Application

This DAG is responsible for deploying the application to the specified environment. It runs a shell script (`deploy_app.sh`) to carry out the deployment and sends email notifications based on the task's success or failure.

### DAG Steps and Explanation

1. **Deploy Application**:
   - `deploy_app`: Executes the `deploy_app.sh` script to deploy the application.

2. **Email Notifications**:
   - `deployment_pass_email`: Sends an email notification on task success, notifying stakeholders of a successful deployment.
   - `deployment_fail_email`: Sends an email notification on task failure, alerting stakeholders to investigate the issue.

3. **Task Dependencies**:
   - The DAG ensures that the application deployment task runs first, followed by sending the success or failure email based on the deployment result.

---

## `deployment_model_dag.py` - DAG for Deploying Model

This DAG is responsible for deploying the trained machine learning model to the specified environment. It runs a shell script (`deploy_model.sh`) to deploy the model and sends email notifications based on the task's success or failure.

### DAG Steps and Explanation

1. **Deploy Model**:
   - `deploy_model`: Executes the `deploy_model.sh` script to deploy the machine learning model.

2. **Email Notifications**:
   - `deployment_pass_email`: Sends an email notification on task success, notifying stakeholders of a successful model deployment.
   - `deployment_fail_email`: Sends an email notification on task failure, alerting stakeholders to investigate the issue.

3. **Task Dependencies**:
   - The DAG ensures that the model deployment task runs first, followed by sending the success or failure email based on the deployment result.

---

## `master_dag.py` - Master DAG for Orchestrating Multiple Tasks

This DAG is responsible for triggering and managing other sub-DAGs related to different stages of data processing, model training, bias detection, and deployment. It coordinates tasks such as bias mitigation, data drift detection, feature importance analysis, model training, bias detection, and deployment.

### DAG Steps and Explanation

1. **Trigger Data Bias Mitigation**:
   - `trigger_data_bias_dag`: Triggers the `bias_detection_and_mitigation` DAG to check for biases in the data and mitigate them if necessary.

2. **Trigger Data Drift Detection**:
   - `trigger_data_drift_dag`: Triggers the `drift_data_dag` to detect if there is any significant drift in the data over time.

3. **Trigger Feature Importance Analysis**:
   - `trigger_feature_imp_dag`: Triggers the `feature_imp_analysis_dag` to analyze the importance of different features in the dataset for model training.

4. **Trigger Model Training**:
   - `trigger_model_train_dag`: Triggers the `model_train_evaluate` DAG for training a model on the processed data.

5. **Trigger Model Bias Detection**:
   - `trigger_model_bias_dag`: Triggers the `model_bias_detection_and_mitigation` DAG to check if the model has any bias and apply mitigation if necessary.

6. **Trigger Model Deployment**:
   - `trigger_model_deploy`: Triggers the `deploy_model_task` DAG to deploy the model after training and validation.

7. **Task Dependencies**:
   - The tasks are arranged in a sequence: Data bias mitigation → Data drift detection and feature importance analysis → Model training → Model bias detection → Model deployment.

---

## `model_bias_detection_and_mitigation_dag.py` - Model Bias Detection and Mitigation DAG

This DAG handles the steps related to detecting and mitigating bias in the machine learning model. It checks if the model has any bias, applies necessary mitigation strategies, and uploads the results to Google Cloud Platform (GCP) for further analysis.

### DAG Steps and Explanation

1. **Load Data Splits**:
   - `load_data_splits`: Loads the training, test, and sensitive feature splits from the dataset to prepare for model training and bias detection.

2. **Perform Bias Detection**:
   - `perform_bias_detection`: Detects bias in the model by analyzing the test data using sensitive features. If bias is detected, the results are saved locally.

3. **Perform Bias Mitigation**:
   - `perform_bias_mitigation`: If bias is detected, this task performs bias mitigation by training a new model and applying techniques to reduce bias in the predictions.

4. **Upload Results to GCP**:
   - `upload_results_to_gcp`: Uploads the results of the bias detection and mitigation process to Google Cloud Platform for storage and further analysis.

5. **Decide to Serve Model**:
   - `proceed_to_serving`: Determines whether the model can be served based on whether bias was mitigated or not.

6. **Push Model to MLflow**:
   - `push_model_to_mlflow`: If the model passes the bias checks and mitigation, it is pushed to MLflow for tracking and serving.

7. **Task Dependencies**:
   - The tasks follow a sequence: Load data splits → Perform bias detection → Perform bias mitigation → Upload results to GCP → Decide on serving the model → Push model to MLflow.

---

## `model_retrain_evaluate_dag.py` - DAG for Model Retraining and Evaluation

This DAG automates the process of retraining a machine learning model and evaluating its performance. It triggers other DAGs related to data collection and model training. The flow ensures that the model is retrained whenever new data becomes available.

### DAG Steps and Explanation

1. **Trigger New Data DAG**:
   - `trigger_new_data_dag`: Triggers the `new_data_dag` to collect new data, which will be used for model retraining. The task waits for the completion of this DAG before proceeding.

2. **Trigger Model Training and Evaluation DAG**:
   - `trigger_model_train_dag`: Triggers the `model_train_evaluate` DAG to train and evaluate the model using the newly collected data. This task also waits for completion before proceeding.

3. **Task Dependencies**:
   - The DAG ensures that the new data collection is completed before initiating the model training and evaluation process.

---

## `model_roll_back_dag.py` - DAG for Rolling Back Model to Previous Version

This DAG automates the rollback of a machine learning model to a previous version in case the current model is performing poorly or encountering issues. It uses a PythonOperator to invoke a rollback function and restore the previous model version in MLflow.

### DAG Steps and Explanation

1. **Run Rollback Task**:
   - `run_rollback_task`: Executes the `run_rollback` function, which retrieves the model name from a JSON configuration file and triggers the rollback process using the `rollback_model` function.

2. **Task Dependencies**:
   - The DAG contains a single task, `run_rollback_task`, which is the only task in this workflow. It is responsible for executing the model rollback logic.

--- 

## `model_train_evaluate_dag.py` - DAG for Training and Evaluating Machine Learning Model

This DAG is responsible for training or fine-tuning a machine learning model on incoming data, evaluating its performance against predefined thresholds, and triggering further actions such as model deployment or retraining based on the evaluation metrics.

### DAG Steps and Explanation

1. **Download Data from DVC**:
   - `download_data_from_dvc_task`: Downloads the required dataset from DVC (Data Version Control) to be used for training or fine-tuning the model.

2. **Task Decision (Train from Scratch or Fine-tune)**:
   - `choose_task`: Decides whether to train the model from scratch or to fine-tune an existing model based on the provided configuration.

3. **Download Model (for Fine-tuning)**:
   - `download_model_task`: Downloads the existing trained model from storage if fine-tuning is needed.

4. **Training the Model (from Scratch)**:
   - `train_on_all_data_task`: If training from scratch is chosen, this task trains the model on the entire dataset downloaded from DVC and saves it locally.

5. **Fine-tuning the Model**:
   - `fine_tune_on_new_data_task`: If fine-tuning is chosen, this task fine-tunes the existing model on the new data and saves the updated model.

6. **Evaluate the Model**:
   - `evaluate_model_task`: Evaluates the trained or fine-tuned model using predefined metrics, such as accuracy, precision, and recall. The evaluation metrics are used for further decision-making.

7. **Threshold Check**:
   - `threshold_check_task`: Checks the evaluation metrics against predefined thresholds. If the thresholds are met, the process proceeds to deployment; otherwise, further actions such as hyperparameter tuning or retraining may be required.

8. **Email Notifications**:
   - `threshold_pass_email`: Sends an email notification when the model meets the required evaluation thresholds.
   - `threshold_fail_email`: Sends an email notification when the model does not meet the evaluation thresholds, indicating the need for further actions.

9. **Delete Local Files**:
   - `delete_local_task`: Cleans up any local data or model files after the process is complete, ensuring that unnecessary files are removed.

10. **Model Deployment**:
    - `trigger_model_deployment`: Triggers the deployment of the model if the evaluation thresholds are met and the model is ready for production.

### Task Dependencies

- The DAG starts by downloading the data, followed by a decision on whether to train from scratch or fine-tune an existing model.
- Depending on the decision, either the training or fine-tuning task is executed.
- The model is evaluated, and based on the evaluation, a threshold check determines whether to deploy the model or initiate further steps such as retraining or hyperparameter tuning.
- After completion, local files are deleted to free up resources, and if the model passes all checks, deployment is triggered.

