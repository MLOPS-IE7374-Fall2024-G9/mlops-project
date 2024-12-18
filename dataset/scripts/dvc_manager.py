import logging
import subprocess
import pandas as pd
import os
import tempfile
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class DVCManager:
    def __init__(self, json_credential_path="mlops-437516-b9a69694c897.json"):
        # Get the directory of the current script
        self.script_dir = os.path.dirname(__file__)  

        # Relative path to the 'data' folder        
        self.data_dir = os.path.join(self.script_dir, "../data")

        # Configure dvc
        self.configure_dvc_credentials(json_credential_path)
        self.all_data_filename = "data_raw.csv"
        self.processed_data_filename = "data_preprocessed.csv"

    def configure_dvc_credentials(self, json_credential_path):
        """
        Function to run the 'dvc remote modify --local' command from Python to set the credential path.
        """
        try:
            # Run the dvc remote modify command
            logger.info(f"Configuring DVC to use credentials from {json_credential_path}.")
            
            subprocess.run(
                [
                    "dvc",
                    "remote",
                    "modify",
                    "--local",
                    "storage",
                    "credentialpath",
                    json_credential_path,
                ],
                check=True
            )

            logger.info("DVC remote configuration successful.")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to configure DVC remote: {e}")

    def upload_data_to_dvc(self, df, file_name):
        """
        Save the DataFrame as a CSV file with a custom name, add it to DVC, and push it to the remote.
        """
        try:
            # Ensure the data directory exists
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)

            # Create a file with the custom filename in the data directory
            temp_file_path = os.path.join(self.data_dir, file_name)
            logger.info(f"Saving DataFrame to CSV file at {temp_file_path}.")
            df.to_csv(temp_file_path, index=False)

            # # Change to the dataset/data directory
            # os.chdir(self.data_dir)

            # Add the file to DVC
            logger.info(f"Adding {file_name} to DVC.")
            subprocess.run(["dvc", "add", temp_file_path], check=True)

            # Push the data to the DVC remote without committing to Git
            logger.info("Pushing dataset to DVC remote (without Git commit).")
            result = subprocess.run(["dvc", "push"], check=False)

            if result.returncode != 0:
                # Log specific error for invalid credentials or failed push
                logger.error(
                    "DVC push failed. Please check your credentials and remote settings."
                )
                return

            # If push succeeds, delete the CSV file
            if os.path.exists(temp_file_path):
                logger.info(f"Deleting CSV file: {temp_file_path}")
                os.remove(temp_file_path)

            logger.info("Dataset uploaded to DVC and CSV file deleted successfully.")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to execute DVC command: {e}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def download_data_from_dvc(self, filename, save_local=0):
        """
        Pull the latest version of the dataset from DVC and return it as a DataFrame.
        """
        try:
            # Pull the latest dataset from DVC
            logger.info("Pulling latest dataset from DVC.")
            subprocess.run(["dvc", "pull", "--force"], check=True)

            # Load the dataset into a DataFrame
            logger.info(f"Loading dataset from {self.data_dir}.")
            csv_files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
            if filename not in csv_files:
                logger.error("No CSV files found in the data directory.")
                return None

            latest_file = os.path.join(self.data_dir, filename)
            df = pd.read_csv(latest_file)

            if save_local == 0:
                logger.info(f"Deleting CSV file in local: {latest_file}")
                for file in csv_files:
                    path = os.path.join(self.data_dir, file)
                    os.remove(path)

            logger.info("Dataset downloaded and loaded into DataFrame successfully.")
            return df, latest_file

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to execute DVC command: {e}")
            return None
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return None
        
    def delete_local_data(self):
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
        logger.info("Deleting CSV file in local")
        for file in csv_files:
            path = os.path.join(self.data_dir, file)
            os.remove(path)

def main():
    parser = argparse.ArgumentParser(description="DVC Manager CLI Tool")
    parser.add_argument("action", choices=["configure", "upload", "download", "delete"], help="Action to perform")
    parser.add_argument("--json_credential_path", default="mlops-437516-b9a69694c897.json", help="Path to DVC JSON credentials")
    parser.add_argument("--file_name", help="File name for upload or download action")
    parser.add_argument("--save_local", type=int, default=0, help="Keep downloaded file locally (1 to save, 0 to delete)")

    args = parser.parse_args()

    dvc_manager = DVCManager(json_credential_path=args.json_credential_path)

    if args.action == "configure":
        dvc_manager.configure_dvc_credentials(args.json_credential_path)
    elif args.action == "upload":
        if args.file_name:
            # Load sample data or replace with actual data for testing
            df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
            dvc_manager.upload_data_to_dvc(df, args.file_name)
        else:
            logger.error("Please specify a file name for the upload action.")
    elif args.action == "download":
        if args.file_name:
            df, file_path = dvc_manager.download_data_from_dvc(args.file_name, args.save_local)
            logger.info(f"Downloaded data file path: {file_path}")
        else:
            logger.error("Please specify a file name for the download action.")
    elif args.action == "delete":
        dvc_manager.delete_local_data()
        logger.info("Local data files deleted successfully.")


if __name__ == "__main__":
    main()