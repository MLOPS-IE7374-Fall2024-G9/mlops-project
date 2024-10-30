import tensorflow_data_validation as tfdv
import pandas as pd
from typing import Optional, Any

class DataSchemaAndStatistics:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.stats: Optional[Any] = None
        self.schema: Optional[Any] = None

    def generate_statistics(self) -> Any:
        """Generates statistics for the dataset."""
        # Generate statistics directly from the DataFrame
        self.stats = tfdv.generate_statistics_from_dataframe(self.data)
        print("Statistics generated successfully.")
        return self.stats

    def infer_schema(self) -> Any:
        """Infers a schema from the dataset statistics."""
        if not self.stats:
            self.generate_statistics()  # Ensure statistics are generated first
        self.schema = tfdv.infer_schema(self.stats)
        print("Schema inferred successfully.")
        return self.schema

    def validate_data(self, new_data: pd.DataFrame, load: bool = False, schema_path: Optional[str] = None) -> Any:
        """Validates new data against the inferred or loaded schema."""
        if load:
            if schema_path is None:
                raise ValueError("Schema path must be provided to load schema.")
            self.load_schema(schema_path)
        elif not self.schema:
            raise ValueError("Schema has not been inferred or loaded. Run infer_schema() or provide a schema to load.")

        # Generate statistics for the new data
        new_stats = tfdv.generate_statistics_from_dataframe(new_data)
        
        # Validate new data against the schema
        anomalies = tfdv.validate_statistics(statistics=new_stats, schema=self.schema)
        
        if anomalies.anomaly_info:
            print("Anomalies found:")
            for anomaly in anomalies.anomaly_info.values():
                print(anomaly)
        else:
            print("No anomalies found. Data is valid.")
        return anomalies

    def save_schema(self, file_path: str):
        """Saves the inferred schema to a specified file path."""
        if not self.schema:
            raise ValueError("Schema has not been inferred. Run infer_schema() first.")
        tfdv.write_schema_text(self.schema, file_path)
        print(f"Schema saved successfully at {file_path}")

    def load_schema(self, file_path: str):
        """Loads a schema from a specified file path."""
        self.schema = tfdv.load_schema_text(file_path)
        print(f"Schema loaded successfully from {file_path}")
