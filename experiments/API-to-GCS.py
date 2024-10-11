import requests
import csv
import os
import tempfile
from google.cloud import storage
from flask import escape

# Function that will be triggered by HTTP requests
def upload_energy_demand(request):
    api_key = "dLxQrZFjnwyM6LfOb9TrNTt5n8n1ulnvjbxsWqNU"  # Replace with your actual API key
    url = f"https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/?api_key={api_key}&frequency=hourly&data[0]=value&start=2024-08-28T00&end=2024-09-29T00&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    
    # Define your GCS bucket name and file path
    bucket_name = 'mlops-project-grp-9'  # Replace with your actual bucket name
    destination_blob_name = 'Electricity_demand/energy_demand.csv'

    # Make the GET request to the API
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()  # Parse the JSON response

        # Use a temporary file to store the CSV before uploading to GCS
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8', delete=False) as temp_file:
            csv_filename = temp_file.name
            writer = csv.writer(temp_file)

            # Write header row
            writer.writerow(["period", "subba", "subba-name", "parent", "parent-name", "value", "value-units"])
            
            # Write data rows
            for record in data['response']['data']:
                writer.writerow([
                    record.get('period'),
                    record.get('subba'),
                    record.get('subba-name'),
                    record.get('parent'),
                    record.get('parent-name'),
                    record.get('value'),
                    record.get('value-units')
                ])

        # Initialize Google Cloud Storage client and upload the file
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        # Upload the CSV file to GCS
        blob.upload_from_filename(csv_filename)

        # Remove the temporary file after upload
        os.remove(csv_filename)

        return f"File uploaded to GCS bucket {bucket_name} as {destination_blob_name}"

    else:
        return f"Failed to fetch data from the API. Status code: {response.status_code}"
