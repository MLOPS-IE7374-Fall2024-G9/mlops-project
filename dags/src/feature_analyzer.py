import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from model.scripts.feature_importance_analyzer import * 

def analyze_features(model_path, data_path):
    analyzer = FeatureImportanceAnalyzer(model_path, data_path)
    analyzer.load_model()
    analyzer.load_data()
    analyzer.compute_shap_values()
    analyzer.plot_shap_summary()