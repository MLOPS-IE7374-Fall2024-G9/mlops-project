import pandas as pd
import pandera as pa
import json
from typing import Optional, Any
import logging

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
        new_data.fillna(0, inplace=True)
        
        if not self.schema:
            raise ValueError("Schema has not been inferred. Run infer_schema() first.")
        
        try:
            validated_data = self.schema.validate(new_data, lazy=True)
            logger.info("New data validated successfully.")
            return 1
        except pa.errors.SchemaErrors as e:
            logger.info(new_data["pressureInches"])
            logger.error("Schema validation errors found:")
            logger.error(e)
            return 0

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
