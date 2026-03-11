import os
import pandas as pd
import joblib
import logging

# Configure standardized logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_info(message):
    logger.info(message)

def log_error(message):
    logger.error(message)

def load_data(data_path='data'):
    """Loads all CSV raw datasets into pandas DataFrames."""
    invoices = pd.read_csv(os.path.join(data_path, 'vendor_invoice.csv'))
    purchases = pd.read_csv(os.path.join(data_path, 'purchases.csv'))
    prices = pd.read_csv(os.path.join(data_path, 'purchase_prices.csv'))
    inventory = pd.read_csv(os.path.join(data_path, 'inventory.csv'))
    return invoices, purchases, prices, inventory

def save_model(model, filename):
    """Saves a model to the models/ directory."""
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, os.path.join('models', filename))
    log_info(f"Model saved: models/{filename}")

def load_model(filename):
    """Loads a model from the models/ directory."""
    try:
        return joblib.load(os.path.join('models', filename))
    except Exception as e:
        log_error(f"Failed to load model {filename}: {e}")
        return None

def save_output(df, filename):
    """Saves analysis output to the outputs/ directory."""
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    df.to_csv(os.path.join('outputs', filename), index=False)
    log_info(f"Output saved: outputs/{filename}")
