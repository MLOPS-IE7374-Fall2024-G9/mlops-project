## Directory Structure

```
data/
├── data_preprocess.csv.dvc       # Preprocessed data file, tracked by DVC
├── data_raw.csv.dvc              # Raw API data file, tracked by DVC

scripts/
├── data_bias.py                  # Script for detecting data bias across specified subsets
├── data_downloader.py            # Script for downloading and saving data from API
├── data_preprocess.py            # Script for data preprocessing (not detailed here)
├── data_schema.py                # Script for data schema validation (not detailed here)
├── data.py                       # Main data module containing classes for data processing
├── dvc_manager.py                # Manages DVC operations for data versioning
├── README.md                     # Documentation for the project
```

---

## File Descriptions

### 1. `data_downloader.py`
This script is responsible for fetching data from an external API for a specified date range and region(s). Users define the date range in `DD-MM-YYYY` format and provide regions in JSON format as command-line arguments. It leverages a `DataCollector` class from `data.py` to interact with the API and retrieve data.

**Logic and Purpose**:
- **Argument Parsing**: Using `argparse`, the script accepts `start_date`, `end_date`, and `regions` as command-line arguments. Dates are validated to be in `DD-MM-YYYY` format, and `regions` is provided as a JSON string, which is parsed into a dictionary.
- **API Interaction**: The `get_updated_data_from_api` function sends requests to the API, providing the specified dates and regions, and returns the resulting data.
- **Data Saving**: The script saves the downloaded data as `downloaded_data.csv` in the current directory.

**Usage**:
```bash
python data_downloader.py --start_date 09-11-2024 --end_date 10-11-2024 --regions '{"new_york": {"ZONEA": [42.8864, -78.8784]}}'
```

### 2. `data_bias.py`
This script detects potential biases in the data by analyzing subsets based on specified columns (e.g., region or category). It enables statistical analysis by slicing data into groups, calculating metrics, and identifying any significant deviations that could indicate bias.

**Logic and Purpose**:
- **Data Slicing**: The `data_slicing` method splits the dataset based on unique values in specified columns. Each subset is logged for reference.
- **Statistics Calculation**: `calculate_statistics` computes the mean value for each subset in the specified metric column.
- **Bias Detection**: Using the `detect_bias` method, the script identifies subsets with significant deviations in mean values compared to the overall dataset, based on a threshold ratio.
- **Report Generation**: The `document_bias_report` method logs a summary of bias findings, indicating any subsets that show potential bias.


### 3. `data_preprocess.py`
This script preprocesses the collected data to prepare it for machine learning or analysis tasks. It includes data cleaning, feature engineering, adding cyclic features, normalization, encoding, and final feature selection.

**Logic and Purpose**:
- **Data Cleaning**: The `clean_data` method removes missing values and duplicates.
- **Feature Engineering**: The `engineer_features` method creates rolling averages, standard deviations, and lag features to capture temporal patterns.
- **Cyclic Features**: The `add_cyclic_features` method adds sine and cosine transformations of the month to capture seasonality.
- **Normalization and Encoding**: The `normalize_and_encode` method normalizes numerical columns and encodes categorical columns.
- **Feature Selection**: The `select_final_features` method retains only the relevant columns for further analysis.
- **DVC Tracking**: The `save_data` method saves the final preprocessed data and tracks it with DVC for version control.

### 4. `data_schema.py`
This script provides schema inference and validation for ensuring data consistency. It uses `pandera` to validate data types, formats, and any custom rules.

**Logic and Purpose**:
- **Schema Inference**: The `infer_schema` method uses `pandera` to automatically infer schema from a dataset.
- **Data Validation**: The `validate_data` method validates new data against the inferred schema, logging any schema errors.
- **Schema Saving and Loading**: The `save_schema` and `load_schema` methods allow the schema to be saved as JSON and loaded when needed.

### 5. `data.py`
This module includes the main data processing classes used in other scripts, such as `DataCollector` and `DataRegions`, for managing data collection and organization.

**Logic and Purpose**:
- **Region Management**: `DataRegions` defines the coordinates for each region to specify where data should be collected.
- **Data Collection**: `DataCollector` manages API calls for demand and weather data, splits dates as required by the API (monthly or yearly), processes weather data, and combines demand and weather data into a single dataset.
- **Date Handling**: Supports splitting dates by month and year, depending on the data requirements of each region.

Currently this project supports data collection for the following regions and sub-regions. Each sub-region is associated with geographical coordinates.
#### Texas
- **COAS**: Houston - [29.749907, -95.358421]
- **EAST**: Tyler - [32.351485, -95.301140]
- **FWES**: Midland - [31.997345, -102.077915]
- **NCEN**: Dallas - [32.78306, -96.80667]
- **NRTH**: Wichita Falls - [33.913708, -98.493387]
- **SCEN**: Austin - [30.267153, -97.743057]
- **SOUT**: McAllen - [26.203407, -98.230012]
- **WEST**: Abilene - [32.448736, -99.733144]

#### New York
- **ZONA**: Buffalo (West) - [42.8864, -78.8784]
- **ZONB**: Rochester (Genesee) - [43.1566, -77.6088]
- **ZONC**: Syracuse (Central) - [43.0481, -76.1474]
- **ZOND**: Plattsburgh (North) - [44.6995, -73.4529]
- **ZONE**: Utica (Mohawk Valley) - [43.1009, -75.2327]
- **ZONF**: Albany (Capital) - [42.6526, -73.7562]
- **ZONG**: Poughkeepsie (Hudson Valley) - [41.7004, -73.9209]
- **ZONH**: Millwood - [41.2045, -73.8176]
- **ZONI**: Yonkers (Dunwoodie) - [40.9439, -73.8674]
- **ZONJ**: New York City (NYC) - [40.7128, -74.0060]
- **ZONK**: Hempstead (Long Island) - [40.7062, -73.6187]

#### New England
- **4001**: Portland, Maine - [43.661471, -70.255326]
- **4002**: Manchester, New Hampshire - [42.995640, -71.454789]
- **4003**: Burlington, Vermont - [44.475882, -73.212072]
- **4004**: Hartford, Connecticut - [41.763710, -72.685097]
- **4005**: Providence, Rhode Island - [41.823989, -71.412834]
- **4006**: New Bedford, Massachusetts (Southeast) - [41.635693, -70.933777]
- **4007**: Springfield, Massachusetts (Western/Central) - [42.101483, -72.589811]
- **4008**: Boston, Massachusetts (Northeast) - [42.358894, -71.056742]

### 6. `dvc_manager.py`
The `DVCManager` class facilitates DVC operations for version control and data management, including uploading, downloading, and managing data files in DVC.

**Logic and Purpose**:
- **DVC Configuration**: Configures DVC with a credential file path using `configure_dvc_credentials`.
- **Upload Data**: The `upload_data_to_dvc` method saves data to a CSV, adds it to DVC, pushes it to the remote, and deletes the local CSV file after a successful push.
- **Download Data**: The `download_data_from_dvc` method pulls the latest dataset from DVC, loads it into a DataFrame, and optionally deletes the local CSV files.
- **Local Data Management**: The `delete_local_data` method deletes any CSV files left locally after DVC operations.
