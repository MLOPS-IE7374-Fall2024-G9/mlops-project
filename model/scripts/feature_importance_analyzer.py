import os
import pickle
import pandas as pd
import numpy as np
import logging
import shap
import matplotlib.pyplot as plt

#logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FeatureImportanceAnalyzer:
    def __init__(self, model_path, data_path):
        
        self.model_path = model_path
        self.data_path = data_path
        self.model = None  # loaded model
        self.data = None   # loaded data
        self.X = None      # features
        self.y = None      # target variable

        logging.info(f"Initialized FeatureImportanceAnalyzer with model_path: {model_path} and data_path: {data_path}")

    def load_model(self):
        try:
            with open(self.model_path, 'rb') as file:
                self.model = pickle.load(file)
            logging.info("Model loaded successfully.")
        except FileNotFoundError:
            logging.error("Error: Model file not found. Check the model path.")
        except Exception as e:
            logging.error(f"An error occurred while loading the model: {e}")

    def load_data(self):
        try:
            self.data = pd.read_csv(self.data_path, delimiter=',')
            target_column = 'value'   #need to change

            # Convert `datetime` column
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            self.data['datetime'] = self.data['datetime'].apply(lambda x: x.timestamp())

            # Categorical columns to numerical codes
            self.data['subba-name'] = self.data['subba-name'].astype('category').cat.codes
            self.data['zone'] = self.data['zone'].astype('category').cat.codes


            self.X = self.data.drop(columns=[target_column])
            self.y = self.data[target_column]
            logging.info("Data loaded and split successfully.")
        except FileNotFoundError:
            logging.error("Error: Data file not found. Check the data path.")
        except Exception as e:
            logging.error(f"An error occurred while loading the data: {e}")

    def compute_shap_values(self):
        if self.model is None or self.X is None:
            logging.error("Model or data not loaded. Cannot compute SHAP values.")
            return

        try:
            if hasattr(self.model, 'feature_names_in_'):
                self.X = self.X[self.model.feature_names_in_]

            # SHAP explainer for tree-based models
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer(self.X)
            logging.info("SHAP values computed successfully.")
        except Exception as e:
            logging.error(f"An error occurred while computing SHAP values: {e}")

    def plot_shap_summary(self, report_filename="shap_summary_plot.png", show_plot=False):
        #SHAP summary plot
        if not hasattr(self, 'shap_values'):
            logging.error("SHAP values not computed. Run `compute_shap_values()` first.")
            return

        try:
            # Determine the current script directory and build the file path
            current_script_dir = os.path.dirname(os.path.realpath(__file__))
            reports_dir = os.path.join(current_script_dir, "../../dags/reports/")
            os.makedirs(reports_dir, exist_ok=True)
            
            # Construct the file path
            plot_path = os.path.join(reports_dir, report_filename)
            
            # Generate and save the SHAP summary plot
            shap.summary_plot(self.shap_values, self.X, show=show_plot)
            plt.savefig(plot_path)
            plt.close()
            
            logging.info(f"SHAP summary plot generated and saved at {plot_path}.")
        except Exception as e:
            logging.error(f"An error occurred while plotting SHAP summary: {e}")


if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.realpath(__file__))

    #Relative paths for model and data
    model_path = os.path.join(current_script_dir, "../../model/pickle/xgboost_model.pkl")
    data_path = os.path.join(current_script_dir, "../../dataset/data/bias_mitigated_data.csv")

    # Instance of the FeatureImportanceAnalyzer class
    analyzer = FeatureImportanceAnalyzer(model_path, data_path)
    analyzer.load_model()
    analyzer.load_data()
    analyzer.compute_shap_values()
    analyzer.plot_shap_summary()

