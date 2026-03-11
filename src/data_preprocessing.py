import pandas as pd
import numpy as np
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from src.utils import load_data, log_info

# Define data validation schema (Industry standard)
invoice_schema = DataFrameSchema({
    "invoice_id": Column(int, nullable=False),
    "vendor_id": Column(str, nullable=False),
    "total_amount": Column(float, Check.ge(0)),
    "freight_cost": Column(float, Check.ge(0)),
    "is_risky": Column(int, Check.isin([0, 1]))
})

def preprocess_data():
    """
    Cleans raw datasets and ensures data quality before modeling.
    """
    try:
        invoices, purchases, prices, inventory = load_data()
        
        # 1. Validation check
        log_info("Starting data validation with Pandera...")
        invoice_schema.validate(invoices)
        log_info("✅ Data validation passed.")

        # 2. Cleaning: handling duplicates and types
        invoices = invoices.drop_duplicates()
        invoices['invoice_date'] = pd.to_datetime(invoices['invoice_date']).dt.date
        
        # 3. Handling missing values
        invoices = invoices.fillna({'total_amount': 0, 'freight_cost': 0})
        
        # Combine data for a comprehensive view
        # We merge invoices with the purchase prices and items
        merged_data = invoices.merge(purchases, on='invoice_id', how='left') \
                               .merge(prices, on='item_id', how='left')
        
        log_info(f"Preprocessed dataset contains {len(merged_data)} rows.")
        return merged_data
        
    except Exception as e:
        log_info(f"❌ Error during preprocessing: {e}")
        return None

if __name__ == "__main__":
    df = preprocess_data()
    if df is not None:
        print(df.head())
