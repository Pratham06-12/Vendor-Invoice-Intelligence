from setuptools import setup, find_packages

setup(
    name="vendor-invoice-intelligence",
    version="0.1.0",
    description="End-to-end Machine Learning for Freight Cost & Invoice Risk Detection",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "joblib",
        "streamlit",
        "xgboost",
        "shap",
        "pandera",
        "duckdb",
        "pyarrow"
    ],
    python_requires=">=3.8",
)
