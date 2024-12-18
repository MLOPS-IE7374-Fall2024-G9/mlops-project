import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dataset.scripts.data_schema import *


# ----------------------------------------------------------
def save_schema(dvc_file_name):
    df = pd.read_csv(dvc_file_name)
    data_schema_obj = DataSchemaAndStatistics(df)
    schema = data_schema_obj.infer_schema()
    data_schema_obj.save_schema(dvc_file_name.split(".")[0] + ".json")


def validate_data(dvc_file_name, api_json):
    df = pd.read_csv(dvc_file_name)
    api_df = pd.read_json(api_json)

    data_schema_obj = DataSchemaAndStatistics(df)
    schema = data_schema_obj.infer_schema()

    valid = data_schema_obj.validate_data(api_df)

    if valid == "None":
        return 0
    else:
        return 1

def fix_anomalies(api_json):
    api_df = pd.read_json(api_json)

    data_schema_obj = DataSchemaAndStatistics(api_df)
    api_df = data_schema_obj.fix_anomalies(api_df)

    json_data_cleaned = api_df.to_json(orient='records', lines=False)
    return json_data_cleaned




