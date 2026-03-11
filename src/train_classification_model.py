import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
import os
from src.utils import save_model, load_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features

def train_classification():
    # 1. Pipeline: Load -> Preprocess -> Engineer
    df_preprocessed = preprocess_data()
    df_engineered = engineer_features(df_preprocessed)
    
    # Selecting numerical features for classification
    # Note: Freight cost can be a feature for risk detection (e.g. unusually high freight)
    X = df_engineered.select_dtypes(include=[np.number]).drop(['invoice_id', 'is_risky'], axis=1)
    y = df_engineered['is_risky']
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Save
    save_model(model, 'risk_detector_classifier.pkl')
    
    # 5. Evaluate (Briefly)
    preds = model.predict(X_test)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print(f"Classification model (Risk Detection) training complete.")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1: {f1:.2f}")
    print(classification_report(y_test, preds))
    
    return model, X_test, y_test

if __name__ == "__main__":
    train_classification()
