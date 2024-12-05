# Energy Demand Forcasting

## Introduction
This project focuses on forecasting energy demand using weather data. The forecasting process is orchestrated using Airflow, allowing for automated data collection, processing, and model execution within a Dockerized MLOps pipeline.

## Setup and reproducibility
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd MLOPS-PROJECT
   ```

2. **Run the setup script** to configure the environment:
   ```bash
   ./setup.sh
   ```

   If the script doesnt work on linux, convert it using the following -
   ```bash
   sudo apt install dos2unix
   dos2unix setup.sh
   ```
   
   Then run it
   ```bash
   ./setup.sh
   ```
   
   This script will:
   - Copy necessary configuration files from the `airflow-config` directory.
   - Create a `.env` file with required environment variables.

4. **Build and start the Docker containers**:
   - Airflow - 
   ```bash
   docker-compose up --build -d
   ```
   This command will build the airflow Docker images and start the containers in detached mode.

   - Backend and Frontend
   ```bash
   ./setup.sh && ./run.sh
   ```

   Backend runs on localhost:8000 and Frontend runs on localhost:8501.
   To check backend API endpoints, navigate to localhost:8000/docs

6. **Access the Airflow UI**:
   - Open your web browser and navigate to [http://localhost:8000/home](http://localhost:8000/home).
   - From here, you can start running the Airflow DAGs to orchestrate the energy demand forecasting workflows.
  
7. **Local Development Env**:
  ```bash
   pip install airflow-config/requirements.txt
   ```

7. **Model Deployment**:
   - Reserve a VM
   - Setup password login on the VM - 
   ```
   sudo nano /etc/ssh/sshd_config
   ```
   Change to - PasswordAuthentication yes

   Next set the password and give root access to the user
   ```
   sudo passwd username
   ```

   You can then ssh using password

   - Setup sshpass (used to login with password)
   ```
   apt-get install sshpass
   apt-get install jq
   ```

   - Install docker in the new VM (https://docs.docker.com/engine/install/debian/)
   - Setup docker access
   ```
   sudo usermod -aG sudo <USERNAME>
   sudo usermod -aG docker <USERNAME>
   newgrp docker
   docker ps
   ```

   - Setup and update the credentials in setup-scripts/config.json
   - Run setup_vm.sh to setup the newly allocated vm
   ```
   ./setup-script/setup_vm.sh
   ```

   Manual Deployment
   - Run deploy_app.sh to deploy and run the model and LLM
   ```
   ./setup-script/deploy_app.sh
   ```
   - Run deploy_model.sh to deploy just the model inside the LLM backend
   ```
   ./setup-script/deploy_model.sh
   ```

   Github Action Deployment
   - Create a pull request with changes to the files - backend/app.py and backend/rag.py and deployment is triggered
  
   Using DAG
   - Run the deployment_model_dag to deploy only the new ML model into the backend container
   - Run the deployment_app_dag to deploy only entire backend LLM with new model inside the backend container

## DAGS


## MLFlow 
mlflow server - http://35.209.190.75:5000/ 

## Repository structure
The repository is organized as follows:

```plaintext
MLOPS-PROJECT/
├── .dvc/                               # DVC configuration and cache for data version control
├── .github/                            # GitHub workflows and actions for CI/CD
├── airflow-config/                     # Configuration files for Airflow (Docker files and requirements.txt)
├── backend/                            # Backend services and APIs
├── dags/                               # Airflow DAGs for orchestrating workflows
├── dataset/                            # Data files and processing scripts
├── docs/                               # Holds documents and images
├── experiments/                        # Experiment tracking and configurations. Contains ipynb files
├── frontend/                           # Frontend services
├── logs/                               # Logs generated during execution
├── model/                              # Trained models and model configurations
├── plugins/                            # Airflow plugins for custom operators and hooks
├── setup-scripts/                      # VM setup and deployment scripts
├── .gitignore                          # Files and directories ignored by Git
├── airflow.cfg                         # Configuration file for Airflow
├── README.md                           # Project documentation
├── encrypt.md                          # Encrypts the gcp dvc mlops config file
├── run.sh                              # Shell script to run backend and frontend containers locally
└── setup.sh                            # Shell script to set up the project environment
```

- All the data related code is present in classes in dataset/scripts.
- All the dag related functionality is present in dags/src. The dags/src imports functionality from dataset/scripts wherever required.
- The data downloaded from dvc is stored in dataset/data/
- The model related code is in model/scripts

## Reproducibility details and data versioning with DVC
- Simply follow the setup to reproduce the entire repo and project.
- DVC init method is mentioned in dataset/data/README.md (scroll down)

## Key Components
### 1. Data Acquisition
   - **Purpose**: This component is responsible for downloading or fetching data from necessary sources. Sources include APIs for demand data collection and weather data collection.
   - **Flow**:
     - Data is collected from external APIs for weather and demand information daily (number of days is configurable).
     - For each run, data from yesterday is fetched hourly for multiple regions across Texas, New York, and New England (dataset/scripts/data.py).
   - **Data Structure**: Raw data contains columns such as `datetime`, `tempF`, `windspeedMiles`, `weatherCode`, `humidity`, etc., along with identifiers for regions and zones.
   -  **Usage**:
      -  There are two scripts that have the data acquisition code.
      -  1) dataset/data.py - This scripts hold the class for DataCollector (methods for API) and DataRegions (data structure for the regions supported)
         2) dataset/data_downloader.py - script uses data.py to download data from region for given dates. Can be used directly on cmd line - python dataset/scripts/data_downloader.py --start_date 09-11-2024 --end_date 10-11-2024 --regions '{"new_york": {"ZONEA": [42.8864, -78.8784]}} - (make sure to install requirements.txt before running this)

### 2. Data Preprocessing
   - **Purpose**: This component includes data cleaning, transformation, and feature engineering steps. The preprocessing code is modular and reusable, allowing for easy adjustments.
   - **Steps**:
     - **Data Cleaning**: Removes missing values and duplicates.
     - **Feature Engineering**: Adds rolling and lagged features to enrich temporal patterns.
     - **Cyclic Features**: Encodes cyclic temporal data (e.g., month) using sine and cosine transformations.
     - **Normalization and Encoding**: Normalizes numerical data and encodes categorical data.
     - **Feature Selection**: Retains only essential features for downstream tasks.
   - **Code Implementation**:
      - Each step has a dedicated function in dataset/scripts/data_preprocess.py
      - dataset/scripts/data_preprocess_script.py is a script to run data preprocessing on bulk data -
        ``` bash
        python data_preprocess_script.py /data/data_raw.csv
        ```

### 3. Test Modules
   - **Purpose**: Ensures robustness and reliability of the pipeline components.
   - **Implementation**:
     - Unit tests are written for each component of the pipeline, especially for data preprocessing and transformation, using frameworks like `pytest`.
     - Integration tests are stored in `dags/src/tests`
   - **Continuous Integration**: Tests are automatically run on every pull request using GitHub Actions.

### 4. Pipeline Orchestration (Airflow DAGs)
   There are multiple dag pipelines implemented

   1)
   - **Purpose**: Manages the entire workflow from data acquisition to final output generation.
   - **Flow**:
     1. Fetch data from the API.
     2. Preprocess the data.
     3. Generate schema with existing data.
     4. Check API data against schema for anomalies.
     5. If anomalies are detected, send an alert email; otherwise continue
     6. Merge API data with DVC-stored historical data.
     7. Push the merged data back to DVC.
     8. Send a success notification email.
   - **Scheduling**: The DAG is scheduled to run everyday to maintain up-to-date data.

   2)
   - **Purpose**: Bulk Preprocess raw data into preprocessed data
   - **Flow**:
     1. Fetch raw collected data from dvc.
     2. Preprocess the data.
     3. Push the merged data back to DVC.
     4. Send a success notification email.
   - **Scheduling**: The DAG is scheduled manually whenever required to preprocess the entire bulk

   3) 
   - **Purpose**: Bias Detection and Mitigation
   - **Flow**:
     1. Fetch preprocessed data from dvc.
     2. Run bias detection
     3. Run bias mitigation
     4. Upload mitigated data to dvc
     5. Send a success notification email
   - **Scheduling**: The DAG is scheduled to run everyday


### 5. Data Versioning with DVC
   - **Purpose**: Ensures that datasets are version-controlled for consistency and reproducibility.
   - **Implementation**:
     - DVC is used to track and manage the version of the dataset, with relevant `.dvc` files included in the repository.
     - The raw data files, stored as CSVs in DVC, contain historical data to facilitate merging with new API data.
  - **Files**:
     - dataset/data/data_raw.csv.dvc, dataset/data/data_preprocess.csv.dvc and dataset/data/bias_mitigated_data.csv.dvc are the three dataset files which are being stored and tracked in dvc.
     - data_raw.csv has raw data from API, with all the columns and without any preprocessing done
     - data_preprocess.csv has the data with 5 steps of preprocessing and selected features as well
     - bias_mitigated_data.csv has data with mitigation of bias within features and zones. This data will be used to feed it directly into the ML model.

### 6. Tracking and Logging
   - **Purpose**: Tracks pipeline progress and monitors for anomalies or errors.
   - **Implementation**:
     - Python’s logging library is used, and logs are stored in the `logs` folder.
     - Airflow’s built-in logging tracks each task in the DAG, making it easy to identify issues and debug.
     - Alerts and monitoring mechanisms are set up to notify via email if anomalies or errors are detected in the dag pipeline.

### 7. Data Schema Generation and Anomaly Detection & Alerts
   - **Purpose**: Automates the generation of data schemas and statistics to monitor data quality over time. Also Identifies data anomalies such as missing values, outliers, or invalid formats and triggers alerts.
   - **Implementation**:
     - For data schema generation and statistics generation, Pandera library is used. It creates a schema from entire dataset and validates new incoming data with respect to this schema. It also generates statistics about the dataset from time to time. 
     - Regular schema validation helps detect and address data quality issues, ensuring that data conforms to expected formats.
     - If no anomalies are detected, the data proceeds through the pipeline without interruptions.

### 8. Pipeline Flow Optimization
   - **Purpose**: Optimization of the pipeline while data preprocessing
   - **Flow**:
      - The dataset is about 120000 rows. It was observed that the data preprocessing per step was very time-consuming.
      - Analyzing the Gantt chart, chunking strategy is applied for raw data to preprocess data generation.
      - There are two chucking strategies applied - 1) chunking based on a number of chunks, and 2) chunking based on weather and demand in geographical regions. These strategies parallelized the flow making the dag run faster.
     
### 9. Bias Detection and Mitigation
   - **Purpose**: Detects and mitigates bias in data by using data slicing and model training techniques.
   - **Implementation**:
     - Data slicing is used to create subsets of data to evaluate performance across different segments.
     - Fairlearn tool is used to implement data slicing. Data is sliced based on "subba-name" which is the region for electricity demand.
     - A bias detection model is also trained to identify potential biases in the data and results, allowing for continuous monitoring of fairness.
    
### 10. Data Drift Detection
- **Purpose**: To identify shifts in the underlying data distribution between the baseline dataset and newly collected data, which can affect model performance. Monitoring data drift helps ensure the model remains accurate and reliable over time.
- **Implementation**:
  - **Evidently AI Data Drift Detection**: Utilizes the DataDriftPreset metric from Evidently AI to generate detailed reports on data drift across various features, both numerical and categorical. This includes HTML and JSON outputs summarizing drift results for analysis.
  - **Kolmogorov-Smirnov (KS) Test**: Applied to numerical features to compare cumulative distribution functions (CDFs) of baseline and new data. The KS test returns a p-value indicating whether drift has occurred.
  - **Population Stability Index (PSI)**: Measures the stability of feature distributions by comparing the proportions of values in specified bins between baseline and new datasets. PSI values above certain thresholds indicate drift.


### Example Data and Processing Flow

The raw dataset contains fields:
datetime, tempF, windspeedMiles, weatherCode, precipMM, precipInches, humidity, visibility, visibilityMiles, pressure, pressureInches, cloudcover, HeatIndexC, HeatIndexF, DewPointC, DewPointF, WindChillC, WindChillF, WindGustMiles, WindGustKmph, FeelsLikeC, FeelsLikeF, uvIndex, subba-name, value, value-units, zone. 

Example row:
2019-06-05T17,82,12,176,0.3,0.0,81,9,5,1008,30,87,32,90,24,76,28,82,23,36,32,90,6,ERCO - Coast,13395,megawatthours,COAS


**Processing Flow**:
   - Data is collected from the API and processed through the pipeline as described, undergoing transformations such as cleaning, feature engineering, normalization, and encoding.
   - Each preprocessing step is handled by a specific function, with data transformations such as cyclic feature addition and rolling means for key columns.

**Region Coordinates for Data Collection**: The DAG fetches data for selected regions across Texas, New York, and New England, capturing energy demand and weather conditions for locations such as Houston, Dallas, New York City, and Boston.

