# SHAP: Based on game theory, SHAP assigns each feature an importance value by measuring its contribution to the prediction. It offers consistent feature importance at both the global and local levels.
# LIME: Perturbs input data and builds an interpretable model locally around a specific prediction to show how features impact that particular outcome.

import pickle
import pandas as pd
import numpy as np
import logging
import shap
import matplotlib.pyplot as plt

# Configure logging
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
            # Align columns to match model's training data
            if hasattr(self.model, 'feature_names_in_'):
                self.X = self.X[self.model.feature_names_in_]

            # Create a SHAP explainer for tree-based models
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer(self.X)
            logging.info("SHAP values computed successfully.")
        except Exception as e:
            logging.error(f"An error occurred while computing SHAP values: {e}")

    def plot_shap_summary(self):
        #SHAP summary plot
        if not hasattr(self, 'shap_values'):
            logging.error("SHAP values not computed. Run `compute_shap_values()` first.")
            return

        try:
            shap.summary_plot(self.shap_values, self.X)
            plt.savefig("shap_summary_plot.png")
            logging.info("SHAP summary plot generated and saved.")
        except Exception as e:
            logging.error(f"An error occurred while plotting SHAP summary: {e}")


if __name__ == "__main__":
    model_path = '/Users/nikhilsirisala/Desktop/Course_Notes/MlOps/mlops-project/model/xgb_reg.pkl'
    data_path = '/Users/nikhilsirisala/Desktop/Course_Notes/MlOps/mlops-project/dataset/data/bias_mitigated_data.csv'

    # Instance of the FeatureImportanceAnalyzer class
    analyzer = FeatureImportanceAnalyzer(model_path, data_path)
    analyzer.load_model()
    analyzer.load_data()
    analyzer.compute_shap_values()
    analyzer.plot_shap_summary()

