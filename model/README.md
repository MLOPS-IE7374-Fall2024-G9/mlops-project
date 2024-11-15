## Structure

```
.
├── data
│   ├── train_data.csv                          # Training dataset
│   ├── validate_data.csv                       # Validation dataset
│   ├── test_data.csv                           # Testing dataset
│
├── experiments                                 # Files for experimentation and model training
│
├── pickle                                      # Folder for saving trained models locally
│
├── scripts
│   ├── config.json                             # Configuration file with model parameters
│   ├── feature_importance_analyzer.py          # Analyzing feature importance
│   ├── data_loader.py                          # Script to load and split datasets
│   ├── train.py                                # Main training script supporting LR, LSTM, and XGBoost
│   ├── utils.py                                # Utility functions, including MLflow integration
│
├── unit_tests.py                               # Unit tests for all scripts

```

## Usage

### Training a Model
You can train a model by running the `train.py` script with the appropriate arguments:

```
python scripts/train.py path/to/dataset.csv --config scripts/config.json --model <model_type>
```

Replace `<model_type>` with one of the following:
- `lr` for Linear Regression
- `lstm` for LSTM
- `xgboost` for XGBoost

### Example Commands

```
python scripts/train.py data/dataset.csv --config scripts/config.json --model lr
python scripts/train.py data/dataset.csv --config scripts/config.json --model lstm
python scripts/train.py data/dataset.csv --config scripts/config.json --model xgboost
```

## Key Features

### 1. MLflow Integration
- Tracks experiments, parameters, metrics, and artifacts for each model.
- Enables loading of saved models from MLflow or local storage.

### 2. Modular Design
- Easily switch between different models (`Linear Regression`, `LSTM`, `XGBoost`).
- Configurable hyperparameters using `config.json`.

### 3. Data Version Control (DVC)
- Tracks dataset versions and ensures reproducibility.

### 4. Utilities
- `utils.py` provides reusable functions for experiment logging and data preprocessing.


## Unit Tests
Run unit tests to validate the functionality of scripts:
```
pytest unit_tests.py
```