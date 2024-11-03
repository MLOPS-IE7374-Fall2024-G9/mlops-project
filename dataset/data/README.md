# Data Version Control (DVC)

This is used to track our changes made in the data files for this project. The actual data itself sits in a Google Cloud bucket. This tool helps us track our data alongside our code. Try not to keep data on Github.

## Getting started
If developing on this project, just install the `requirements_dev.txt` by running `pip install -r requirements_dev.txt` AFTER activating your development environment. It should install the DVC package (along with all other dependencies).

## Configure Google Cloud Credentials Locally
You need this for DVC to get and upload data from our data bucket on GCP. Follow these instructions. <br>
1. Ask Samanvya for the json which contains credentials for Google Cloud Service Account.
2. Keep the file somewhere in the system then run: <br>
`dvc remote modify --local storage credentialpath /path/to/json`
3. This should be enough, if not, setup [gcloud auth](https://cloud.google.com/sdk/docs/initializing) or ask Samanvya.

## Get Latest Data
If your dataset/data folder is empty, that's a good sign for you to pull the latest data that this project depends on. You'll have to install DVC first, after that simply run:<br>
`dvc pull` <br> and it'll fetch the latest data from our secure google cloud bucket to the `dataset/` folder. Any issues pulling data could be from not configuring your Google Cloud credentials as described earlier.

## Push Latest Data
As a rule of thumb, put all datafiles in `dataset/data` folder with descriptive names. 
### If you have data files that you want to add to version control:
1. Run `dvc add /path/to/data_file`
<br>This step is similar to adding a code file to git. It creates a new 'dvc' file with your new data file as it's name. It just tracks metadata about the file, also adds the data file to `.gitignore` so it doesn't push to git. 

2. Add the previously mentioned two files to git. DVC should also return the git command to add the two new changes to git. Command will change as per your data file.
`git add dataset\data\file_name.csv.dvc dataset\.gitignore` <br> followed by a git commit <br>
`git commit -m "Added {your data file desc}"`

3. Run `dvc config core.autostage true`
<br> This step just automatically stages your data changes but doesn't commit them. That's the next step.

3. Run `dvc push` to push this latest data to the already configured google cloud bucket. If you run into errors, you probably did not setup the google cloud credentials. Probably ask Aakash or Samanvya.

### If you want to modify existing dataset already tracked by DVC

Let's say you modified a dataset file (added more rows or features), simply follow these steps so your teammates can get all the latest data as well.

1. `dvc commit` <br> This commits all changes made to existing datasets already tracked by DVC. To track new datasets, check the previous subheading.
2. `git add dataset/data/*.dvc && git commit -m "Edited {data change details}` <br> This commits any DVC files that have been changed by you.
3. `dvc push` pushes data to cloud bucket.
4. `git push...` commit your changes to git pertaining to the DVC files


# Dataset 

## 1. Dataset Introduction

The dataset comprises hourly weather data, population density, and electricity demand from multiple U.S. regions over the past five years. The primary goal is to predict short-term electricity consumption based on real-time and historical weather conditions, population, and other external factors. By building a comprehensive dataset, this project aims to model the correlation between these factors and electricity demand, helping to optimize grid management.

## 2. Data Card

- **Size**: Estimated ~40,000 data points per region, with hourly data for 5 years across multiple cities.
- **Format**: CSV (processed from API responses)

### Features
- **Weather**: Hourly data including temperature, humidity, wind speed, pressure, etc.
- **Population Density**: High-resolution data (~1 km grid) for population density.
- **Location**: latitude, longitude.
- **Time**: Hourly timestamps over a 5-year period.
- **Label**: Electricity demand (measured in MWh).

### Data Types
- **Numerical**: Temperature, wind speed, population density, electricity demand.
- **Categorical**: Location Zones.

## 3. Data Sources

- **Weather Data**: World Weather Online API offers detailed weather data for different cities, including temperature, wind speed, humidity, etc., on an hourly basis (World Weather Online).
- **Electricity Demand Data**: EIA API provides access to real-time and historical electricity consumption data for different regions across the U.S. We are focusing on Texas - ERCO (ERCOT - Electric Reliability Council of Texas) - which has 8 regions within it covering the entire Texas state. This can be expanded to other regions and states.
- **Population Density Data**: WorldPop API delivers population density data at high resolution (1 km grid), adjusted to match UN estimates, allowing us to account for population factors affecting electricity consumption (WorldPop).

## 4. Data Rights and Privacy

All data from the above sources is publicly available and can be accessed through APIs under open usage terms. WorldPop data is licensed under Creative Commons Attribution 4.0 International, which allows for redistribution and adaptation, provided proper attribution is given. There are no privacy concerns as the dataset does not include personally identifiable information (PII), and it aligns with data protection standards such as GDPR.

---

## Raw Data Sample
| datetime         | tempF | windspeedMiles | weatherCode | precipMM | precipInches | humidity | visibility | visibilityMiles | pressure | pressureInches | cloudcover | HeatIndexC | HeatIndexF | DewPointC | DewPointF | WindChillC | WindChillF | WindGustMiles | WindGustKmph | FeelsLikeC | FeelsLikeF | uvIndex | subba-name     | value | value-units    | zone |
|------------------|-------|----------------|-------------|----------|--------------|----------|------------|-----------------|----------|----------------|------------|------------|------------|-----------|-----------|------------|------------|----------------|--------------|------------|------------|---------|----------------|-------|----------------|------|
| 2019-01-01T00    | 36    | 11             | 356         | 4.2      | 0.2          | 89       | 5          | 3               | 1014     | 30             | 100        | 2          | 36         | 2         | 35        | -2         | 29         | 28             | 44           | -2         | 29         | 1       | ISNE - Maine   | 1509  | megawatthours  | 4001 |

---

## Preprocessed Data Sample
| datetime            | precipMM            | weatherCode   | visibility | HeatIndexF     | WindChillF     | windspeedMiles | FeelsLikeF | tempF_rolling_mean | windspeedMiles_rolling_mean | humidity_rolling_mean | value          | pressure | pressureInches | cloudcover | uvIndex | tempF_rolling_std | windspeedMiles_rolling_std | humidity_rolling_std | tempF_lag_2       | windspeedMiles_lag_2 | humidity_lag_2      | tempF_lag_4      | windspeedMiles_lag_4 | humidity_lag_4       | tempF_lag_6      | windspeedMiles_lag_6 | humidity_lag_6     | month_sin  | month_cos  | subba-name          | zone |
|---------------------|---------------------|---------------|------------|----------------|----------------|----------------|------------|--------------------|-----------------------------|-----------------------|-----------------|----------|----------------|------------|---------|-------------------|---------------------------|----------------------|-------------------|-----------------------|----------------------|------------------|-----------------------|----------------------|------------------|-----------------------|-------------------|-------------|-------------|----------------------|------|
| 2019-01-01 00:00:00 | 0.0527              | 0.8617        | 1.0        | 0.4029         | 0.4575         | 0.2            | 0.4375     | 0.4226             | 0.3270                      | 0.9229                | 0.0520          | 0.5616   | 0.5            | 1.0        | 0.0     | 0.1419            | 0.0898                    | 0.1097               | 0.4773            | 0.2                   | 0.9583               | 0.4621           | 0.25                  | 0.9167               | 0.4167           | 0.275                 | 0.8854           | 0.75        | 0.9330      | ISNE - New Hampshire | 4002 |