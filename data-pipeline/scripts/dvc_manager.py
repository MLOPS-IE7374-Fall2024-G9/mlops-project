import logging
import subprocess
import pandas as pd
import os
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class DVCManager:
    def __init__(self, json_credential_path):
        # Set the data directory relative to the current script's location
        script_dir = os.path.dirname(__file__)  # Get the directory of the current script
        self.data_dir = os.path.join(script_dir, "../data")  # Relative path to the 'data' folder
        self.configure_dvc_credentials(json_credential_path)
        self.filename = "demand_weather_data.csv"

    def configure_dvc_credentials(self, json_credential_path):
        """
        Function to run the 'dvc remote modify --local' command from Python to set the credential path.
        """
        try:
            # Run the dvc remote modify command
            logger.info(f"Configuring DVC to use credentials from {json_credential_path}.")
            subprocess.run(
                ["dvc", "remote", "modify", "--local", "storage", "credentialpath", json_credential_path],
                check=True
            )
            logger.info("DVC remote configuration successful.")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to configure DVC remote: {e}")