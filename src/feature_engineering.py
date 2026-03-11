import pandas as pd
import numpy as np
from src.data_preprocessing import preprocess_data

def engineer_features(df):
    """
    Apply SQL-like feature engineering:
    - Vendor History: Average amount per vendor
    - Seasonality: Month of invoice
    - Item complexity: Unit prices, quantities
    - Discrepancy (synthetic example)
    """
    
    # 1. Vendor performance features
    vendor_stats = df.groupby('vendor_id')['total_amount'].agg(['mean', 'std']).reset_index()
    vendor_stats.columns = ['vendor_id', 'vendor_avg_amt', 'vendor_std_amt']
    df = pd.merge(df, vendor_stats, on='vendor_id', how='left')
    
    # 2. Time-based features
    df['invoice_month'] = pd.to_datetime(df['invoice_date']).dt.month
    df['invoice_dow'] = pd.to_datetime(df['invoice_date']).dt.weekday
    
    # 3. Item-related features
    df['item_total_val'] = df['quantity'] * df['unit_price']
    
    # 4. Binary/Categorical encoding (Mock for simple ML models)
    df = pd.get_dummies(df, columns=['vendor_id'], prefix='v_')
    
    # Drops for feature set
    # Note: Keep invoice_id for mapping, but remove date/non-numeric for simple model input
    df_features = df.copy()
    
    print("Feature engineering completed.")
    return df_features

if __name__ == "__main__":
    df_preprocessed = preprocess_data()
    df_engineered = engineer_features(df_preprocessed)
    print(df_engineered.head())
    df_engineered.to_csv('data/engineered_data.csv', index=False)
