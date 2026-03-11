import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from src.utils import load_model, log_info

def explain_model(model_name='freight_cost_regressor.pkl', output_path='outputs'):
    """
    Generates SHAP values for model transparency.
    Shows which features contribute most to predictions.
    Industy standard for Explainable AI (XAI).
    """
    try:
        model = load_model(model_name)
        # Assuming engineered_data.csv exists for demonstration
        if not os.path.exists('data/engineered_data.csv'):
            log_info("Engineered data not found. Please run feature_engineering.py first.")
            return
            
        df = pd.read_csv('data/engineered_data.csv')
        
        # Selecting numeric sample for SHAP
        X = df.select_dtypes(include=[np.number]).drop(['invoice_id', 'freight_cost', 'is_risky'], axis=1, errors='ignore').iloc[:200]
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Plot and save summary
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, show=False)
        plt.title(f'SHAP Interpretability: {model_name}')
        plt.tight_layout()
        
        if not os.path.exists(output_path): os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, f'shap_summary_{model_name.replace(".pkl","")}.png'))
        plt.close()
        
        log_info(f"✅ SHAP explanation saved for {model_name}.")

    except Exception as e:
        log_info(f"Error during SHAP calculation: {e}")

if __name__ == "__main__":
    explain_model('freight_cost_regressor.pkl')
    explain_model('risk_detector_classifier.pkl')
