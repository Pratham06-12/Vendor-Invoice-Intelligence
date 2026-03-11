import pandas as pd
import numpy as np
import os

def generate_synthetic_data(data_path='data'):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    np.random.seed(42)
    n_samples = 1000
    
    # 1. Vendor Invoices
    vendors = ['Vendor_A', 'Vendor_B', 'Vendor_C', 'Vendor_D', 'Vendor_E']
    invoice_data = {
        'invoice_id': range(1, n_samples + 1),
        'vendor_id': np.random.choice(vendors, n_samples),
        'invoice_date': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, n_samples), unit='d'),
        'total_amount': np.random.uniform(500, 50000, n_samples),
        'freight_cost': np.nan,  # To be predicted
        'is_risky': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]) # 15% risk rate
    }
    
    # Create regression target (Freight Cost) with some logic
    # Freight cost ~ 2-8% of total_amount + some random noise
    invoice_data['freight_cost'] = invoice_data['total_amount'] * np.random.uniform(0.02, 0.08, n_samples) + np.random.normal(50, 20, n_samples)
    
    df_invoices = pd.DataFrame(invoice_data)
    df_invoices.to_csv(os.path.join(data_path, 'vendor_invoice.csv'), index=False)
    
    # 2. Purchases
    purchase_data = {
        'purchase_id': range(1, n_samples + 1),
        'invoice_id': range(1, n_samples + 1),
        'item_id': np.random.randint(100, 200, n_samples),
        'quantity': np.random.randint(1, 100, n_samples),
        'order_date': df_invoices['invoice_date'] - pd.to_timedelta(np.random.randint(1, 15, n_samples), unit='d')
    }
    df_purchases = pd.DataFrame(purchase_data)
    df_purchases.to_csv(os.path.join(data_path, 'purchases.csv'), index=False)
    
    # 3. Purchase Prices
    price_data = {
        'item_id': range(100, 200),
        'unit_price': np.random.uniform(10, 500, 100)
    }
    df_prices = pd.DataFrame(price_data)
    df_prices.to_csv(os.path.join(data_path, 'purchase_prices.csv'), index=False)
    
    # 4. Inventory
    inventory_data = {
        'item_id': range(100, 200),
        'stock_level': np.random.randint(0, 1000, 100),
        'reorder_point': np.random.randint(50, 200, 100)
    }
    df_inventory = pd.DataFrame(inventory_data)
    df_inventory.to_csv(os.path.join(data_path, 'inventory.csv'), index=False)
    
    print(f"Synthetic data generated in {data_path}/")

if __name__ == "__main__":
    generate_synthetic_data()
