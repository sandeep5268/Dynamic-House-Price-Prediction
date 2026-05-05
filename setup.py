from setuptools import setup, find_packages

setup(
    name = "houseprice",
    version = "0.1",
    packages = find_packages(),
    install_requires = [
        "pandas>=1.3",
        "numpy>=1.21",
        "scikit-learn>=1.0",
        "matplotlib>=3.5",
        "seaborn>=0.11",
        "fastapi>=0.95",
        "uvicorn>=0.20",
        "joblib>=1.1",
        "python-multipart>=0.0.5",
        "xgboost"
    ]
)