import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os
from src.utils import load_model, load_data, save_output
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features

def evaluate_all():
    # 1. Pipeline: Prep Data
    df_preprocessed = preprocess_data()
    df_engineered = engineer_features(df_preprocessed)
    
    # 2. Regression Evaluation
    reg_model = load_model('freight_cost_regressor.pkl')
    X_reg = df_engineered.select_dtypes(include=[np.number]).drop(['invoice_id', 'freight_cost', 'is_risky'], axis=1)
    y_reg = df_engineered['freight_cost']
    reg_preds = reg_model.predict(X_reg)
    
    mae = mean_absolute_error(y_reg, reg_preds)
    rmse = np.sqrt(mean_squared_error(y_reg, reg_preds))
    r2 = r2_score(y_reg, reg_preds)
    
    reg_metrics = pd.DataFrame({'Metric': ['MAE', 'RMSE', 'R2'], 'Value': [mae, rmse, r2]})
    save_output(reg_metrics, 'regression_metrics.csv')
    
    # 3. Classification Evaluation
    cls_model = load_model('risk_detector_classifier.pkl')
    X_cls = df_engineered.select_dtypes(include=[np.number]).drop(['invoice_id', 'is_risky'], axis=1)
    y_cls = df_engineered['is_risky']
    cls_preds = cls_model.predict(X_cls)
    
    report = classification_report(y_cls, cls_preds, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    save_output(df_report, 'classification_report.csv')
    
    # Visualizations
    save_plots(y_reg, reg_preds, y_cls, cls_preds)
    
    print("Evaluation complete. Check 'outputs/' directory.")

def save_plots(y_reg, reg_preds, y_cls, cls_preds):
    if not os.path.exists('outputs'): os.makedirs('outputs')
    
    # Plot 1: Regression (Actual vs Pred)
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=y_reg, y=reg_preds)
    sns.lineplot(x=[0, max(y_reg)], y=[0, max(y_reg)], color='red')
    plt.title('Freight Cost: Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig('outputs/regression_actual_vs_pred.png')
    plt.close()
    
    # Plot 2: Confusion Matrix
    cm = confusion_matrix(y_cls, cls_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Invoice Risk: Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('outputs/confusion_matrix_risk.png')
    plt.close()

if __name__ == "__main__":
    evaluate_all()
