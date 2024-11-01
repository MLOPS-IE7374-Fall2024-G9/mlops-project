# Energy Demand Forcasting

## Introduction
This project focuses on forecasting energy demand using weather data. The forecasting process is orchestrated using Airflow, allowing for automated data collection, processing, and model execution within a Dockerized MLOps pipeline.

## Setup
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd MLOPS-PROJECT
   ```

2. **Run the setup script** to configure the environment:
   ```bash
   ./setup.sh
   ```

   This script will:
   - Copy necessary configuration files from the `airflow-config` directory.
   - Create a `.env` file with required environment variables.

3. **Build and start the Docker containers**:
   ```bash
   docker-compose up --build -d
   ```

   This command will build the Docker images and start the containers in detached mode.

4. **Access the Airflow UI**:
   - Open your web browser and navigate to [http://localhost:8000/home](http://localhost:8000/home).
   - From here, you can start running the Airflow DAGs to orchestrate the energy demand forecasting workflows.


## Repository structure
The repository is organized as follows:

```plaintext
MLOPS-PROJECT/
├── .dvc/                     # DVC configuration and cache for data version control
├── .github/                  # GitHub workflows and actions for CI/CD
├── airflow-config/           # Configuration files for Airflow (Docker files and requirements.txt)
├── backend/                  # Backend services and APIs
├── config/                   # Configuration files and settings for Airflow container
├── dags/                     # Airflow DAGs for orchestrating workflows
├── dataset/                  # Data files and processing scripts
├── experiments/              # Experiment tracking and configurations
├── frontend/                 # Frontend services
├── logs/                     # Logs generated during execution
├── model/                    # Trained models and model configurations
├── plugins/                  # Airflow plugins for custom operators and hooks
├── rag/                      # Repository for RAG (Retrieval-Augmented Generation)
├── .env                      # Environment variables for sensitive configurations
├── .gitignore                # Files and directories ignored by Git
├── airflow.cfg               # Configuration file for Airflow
├── README.md                 # Project documentation
└── setup.sh                  # Shell script to set up the project environment
```

## Key Components

### 1. Data Acquisition
   - **Purpose**: This component is responsible for downloading or fetching data from necessary sources, such as APIs and databases, ensuring reproducibility by specifying dependencies in `requirements.txt` or `environment.yml`.
   - **Flow**:
     - Data is collected from external APIs for weather and demand information every 7 days.
     - For each run, data for the last 7 days is fetched hourly for multiple regions across Texas, New York, and New England.
   - **Data Structure**: Raw data contains columns such as `datetime`, `tempF`, `windspeedMiles`, `weatherCode`, `humidity`, etc., along with identifiers for regions and zones.

### 2. Data Preprocessing
   - **Purpose**: This component includes data cleaning, transformation, and feature engineering steps. The preprocessing code is modular and reusable, allowing for easy adjustments.
   - **Steps**:
     - **Data Cleaning**: Removes missing values and duplicates.
     - **Feature Engineering**: Adds rolling and lagged features to enrich temporal patterns.
     - **Cyclic Features**: Encodes cyclic temporal data (e.g., month) using sine and cosine transformations.
     - **Normalization and Encoding**: Normalizes numerical data and encodes categorical data.
     - **Feature Selection**: Retains only essential features for downstream tasks.
   - **Code Implementation**: Each step has a dedicated function in `dags/src`, making the code modular and easy to test.

### 3. Test Modules
   - **Purpose**: Ensures robustness and reliability of the pipeline components.
   - **Implementation**:
     - Unit tests are written for each component of the pipeline, especially for data preprocessing and transformation, using frameworks like `pytest`.
     - Tests are stored in `dags/src/tests` (for DAG functions) and `dataset/tests` (for dataset script functions).
     - Tests cover edge cases, missing values, and potential anomalies.
   - **Continuous Integration**: Tests are automatically run on every pull request using GitHub Actions.

### 4. Pipeline Orchestration (Airflow DAGs)
   - **Purpose**: Manages the entire workflow from data acquisition to final output generation.
   - **Flow**:
     1. Fetch data from the API.
     2. Preprocess the data.
     3. Generate and validate data schema.
     4. Check API data against schema for anomalies.
     5. If anomalies are detected, send an alert email; otherwise:
         - Merge API data with DVC-stored historical data.
         - Push the merged data back to DVC.
         - Send a success notification email.
   - **Scheduling**: The DAG is scheduled to run every 7 days to maintain up-to-date data.

### 5. Data Versioning with DVC
   - **Purpose**: Ensures that datasets are version-controlled for consistency and reproducibility.
   - **Implementation**:
     - DVC is used to track and manage the version of the dataset, with relevant `.dvc` files included in the repository.
     - The raw data files, stored as CSVs in DVC, contain historical data to facilitate merging with new API data.

### 6. Tracking and Logging
   - **Purpose**: Tracks pipeline progress and monitors for anomalies or errors.
   - **Implementation**:
     - Python’s logging library is used, and logs are stored in the `logs` folder.
     - Airflow’s built-in logging tracks each task in the DAG, making it easy to identify issues and debug.
     - Alerts and monitoring mechanisms are set up to notify via email if anomalies or errors are detected in the pipeline.

### 7. Data Schema & Statistics Generation
   - **Purpose**: Automates the generation of data schemas and statistics to monitor data quality over time.
   - **Implementation**:
     - Tools like ML Metadata (MLMD) or TensorFlow Data Validation (TFDV) are used to define and validate data schemas.
     - Regular schema validation helps detect and address data quality issues, ensuring that data conforms to expected formats.

### 8. Anomaly Detection & Alerts
   - **Purpose**: Identifies data anomalies such as missing values, outliers, or invalid formats and triggers alerts.
   - **Flow**:
     - The API data is checked against the schema. If anomalies are found, an alert email is sent.
     - If no anomalies are detected, the data proceeds through the pipeline without interruptions.
   - **Alerts**: Notifications are sent via email or Slack when anomalies are detected.

### 9. Pipeline Flow Optimization
   - **Purpose**: Identifies and addresses bottlenecks in the pipeline.
   - **Implementation**:
     - Airflow’s Gantt chart is used to monitor task duration and detect slow-performing tasks.
     - Tasks are optimized by parallelizing or improving performance as needed.

### 10. Bias Detection
   - **Purpose**: Detects and mitigates bias in data by using data slicing and model training techniques.
   - **Implementation**:
     - Data slicing is used to create subsets of data to evaluate performance across different segments.
     - A bias detection model is trained to identify potential biases in the data and results, allowing for continuous monitoring of fairness.

### Example Data and Processing Flow

**Sample Data**: The dataset contains fields such as `datetime`, `tempF`, `windspeedMiles`, `weatherCode`, `humidity`, and more. Example row:
```
2019-06-05T17,82,12,176,0.3,0.0,81,9,5,1008,30,87,32,90,24,76,28,82,23,36,32,90,6,ERCO - Coast,13395,megawatthours,COAS
```

**Processing Flow**:
   - Data is collected from the API and processed through the pipeline as described, undergoing transformations such as cleaning, feature engineering, normalization, and encoding.
   - Each preprocessing step is handled by a specific function, with data transformations such as cyclic feature addition and rolling means for key columns.

**Region Coordinates for Data Collection**: The DAG fetches data for selected regions across Texas, New York, and New England, capturing energy demand and weather conditions for locations such as Houston, Dallas, New York City, and Boston.

