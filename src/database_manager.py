import duckdb
import pandas as pd
import os

class DatabaseManager:
    """
    Demonstrates professional SQL integration in an ML pipeline.
    Uses DuckDB for high-performace SQL transformations on pandas DataFrames.
    """
    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.conn = duckdb.connect(database=':memory:') # In-memory database
        self._load_csv_to_db()

    def _load_csv_to_db(self):
        """Loads all raw CSV files as SQL tables."""
        for file in os.listdir(self.data_path):
            if file.endswith('.csv'):
                table_name = file.replace('.csv', '')
                file_path = os.path.join(self.data_path, file)
                # Load CSV into DuckDB via SQL
                self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path.replace('\\', '/')}')")
        
        print("SQL Database Manager: All raw tables successfully loaded into DuckDB.")

    def run_query(self, query):
        """Helper to run any SQL query and return a DataFrame."""
        return self.conn.execute(query).df()

    def get_feature_set(self):
        """
        Executes a complex SQL JOIN and aggregation to create the base feature set.
        Demonstrates SQL skills: CTEs, Window Functions, and Multiple Joins.
        """
        query = """
        WITH vendor_metrics AS (
            SELECT 
                vendor_id,
                AVG(total_amount) OVER(PARTITION BY vendor_id) as avg_vendor_spend,
                COUNT(invoice_id) OVER(PARTITION BY vendor_id) as total_vendor_invoices,
                STDDEV(total_amount) OVER(PARTITION BY vendor_id) as std_vendor_spend
            FROM vendor_invoice
        ),
        purchase_details AS (
            SELECT 
                p.invoice_id,
                p.item_id,
                p.quantity,
                pp.unit_price,
                (p.quantity * pp.unit_price) as calc_total
            FROM purchases p
            JOIN purchase_prices pp ON p.item_id = pp.item_id
        )
        SELECT 
            vi.*,
            vm.avg_vendor_spend,
            vm.total_vendor_invoices,
            vm.std_vendor_spend,
            pd.item_id,
            pd.quantity,
            pd.unit_price,
            pd.calc_total,
            CASE WHEN vi.total_amount > (vm.avg_vendor_spend + 2 * vm.std_vendor_spend) THEN 1 ELSE 0 END as is_outlier_amount
        FROM vendor_invoice vi
        JOIN vendor_metrics vm ON vi.vendor_id = vm.vendor_id
        LEFT JOIN purchase_details pd ON vi.invoice_id = pd.invoice_id
        """
        return self.run_query(query)

if __name__ == "__main__":
    db = DatabaseManager()
    features = db.get_feature_set()
    print("Features derived via SQL:")
    print(features.head())
    # Save for downstream use
    if not os.path.exists('data/processed'): os.makedirs('data/processed')
    features.to_csv('data/processed/sql_derived_features.csv', index=False)
