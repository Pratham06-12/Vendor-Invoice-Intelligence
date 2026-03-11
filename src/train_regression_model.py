import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os
from src.utils import save_model, load_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features

def train_regression():
    # 1. Pipeline: Load -> Preprocess -> Engineer
    df_preprocessed = preprocess_data()
    df_engineered = engineer_features(df_preprocessed)
    
    # Selecting numerical features only for regression
    X = df_engineered.select_dtypes(include=[np.number]).drop(['invoice_id', 'freight_cost', 'is_risky'], axis=1)
    y = df_engineered['freight_cost']
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Save
    save_model(model, 'freight_cost_regressor.pkl')
    
    # 5. Evaluate (Briefly)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"Regression model (Freight Cost) training complete.")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}")
    
    return model, X_test, y_test

if __name__ == "__main__":
    train_regression()
