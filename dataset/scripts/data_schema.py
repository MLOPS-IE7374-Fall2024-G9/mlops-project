import pandas as pd
import pandera as pa
import json
from typing import Optional, Any
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

class DataSchemaAndStatistics:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.schema: Optional[pa.DataFrameSchema] = None

    def infer_schema(self) -> pa.DataFrameSchema:
        """
        Infers a schema from the dataset based on column data types.
        """
        # Automatically infer the schema based on the data's structure
        self.schema = pa.infer_schema(self.data)
        logger.info("Schema inferred automatically from dataset.")
        return self.schema

    def validate_data(self, new_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Validates new data against the inferred schema.
        """
        
        if not self.schema:
            raise ValueError("Schema has not been inferred. Run infer_schema() first.")
        
        try:
            validated_data = self.schema.validate(new_data, lazy=True)
            logger.info("New data validated successfully.")
            return 1
        except pa.errors.SchemaErrors as e:
            new_data.fillna(0, inplace=True)
            logger.error("Schema validation errors found:")
            logger.error(e)
            return 0
        
    def fix_anomalies(self, df: pd.DataFrame) -> None:
        """
        Identifies and fixes common data anomalies.
        """
        # Fix missing values
        df.fillna(0, inplace=True)  # Replace NaNs with 0
        
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        
        # Ensure no negative values for specific columns
        columns_to_check = [
            'precipMM', 'visibility', 'HeatIndexF', 'WindChillF', 'windspeedMiles',
            'FeelsLikeF', 'tempF_rolling_mean', 'windspeedMiles_rolling_mean', 
            'humidity_rolling_mean', 'pressure', 'pressureInches', 'cloudcover', 
            'uvIndex', 'tempF_rolling_std', 'windspeedMiles_rolling_std', 
            'humidity_rolling_std', 'tempF_lag_2', 'tempF_lag_4', 'tempF_lag_6', 
            'windspeedMiles_lag_2', 'windspeedMiles_lag_4', 'windspeedMiles_lag_6', 
            'humidity_lag_2', 'humidity_lag_4', 'humidity_lag_6', 'value'
        ]
        for col in columns_to_check:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if x >= 0 else 0)
        
        logger.info("Anomalies fixed successfully.")

        return df

    def save_schema(self, file_path: str):
        """
        Saves the inferred schema to a specified file path in JSON format.
        """
        if not self.schema:
            raise ValueError("Schema has not been inferred. Run infer_schema() first.")
        
        # Convert schema to a dictionary and save it as JSON
        schema_dict = self.schema.to_json()
        with open(file_path, 'w') as file:
            json.dump(schema_dict, file, indent=4)
        logger.info(f"Schema saved successfully at {file_path}")

    def load_schema(self, file_path: str):
        """
        Loads a schema from a specified file path in JSON format.
        """
        with open(file_path, 'r') as file:
            schema_dict = json.load(file)
        self.schema = pa.DataFrameSchema.from_json(schema_dict)
        logger.info(f"Schema loaded successfully from {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Data Schema and Statistics Tool")
    parser.add_argument("--file", type=str, help="Path to the CSV file to load data", required=True)
    parser.add_argument("--infer-schema", action="store_true", help="Infer schema from data")
    parser.add_argument("--validate-data", type=str, help="Path to new data CSV file for validation")
    parser.add_argument("--fix-anomalies", action="store_true", help="Fix anomalies in the data")
    parser.add_argument("--save-schema", type=str, help="File path to save inferred schema as JSON")
    parser.add_argument("--load-schema", type=str, help="File path to load schema from JSON")
    
    args = parser.parse_args()

    # Load initial data
    data = pd.read_csv(args.file)
    tool = DataSchemaAndStatistics(data)

    # Infer schema
    if args.infer_schema:
        tool.infer_schema()

    # Load schema if specified
    if args.load_schema:
        tool.load_schema(args.load_schema)

    # Validate new data if provided
    if args.validate_data:
        new_data = pd.read_csv(args.validate_data)
        tool.validate_data(new_data)

    # Fix anomalies in the loaded data if specified
    if args.fix_anomalies:
        tool.fix_anomalies(data)

    # Save schema to file if specified
    if args.save_schema:
        tool.save_schema(args.save_schema)

if __name__ == "__main__":
    main()

# usage - python script_name.py --file data.csv --infer-schema --save-schema schema.json --validate-data new_data.csv
