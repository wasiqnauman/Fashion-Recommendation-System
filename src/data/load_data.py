"""Utilities for loading raw recommendation datasets."""

from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path("data/raw")


def _load_csv(file_path: Path) -> pd.DataFrame:
    """ Load a CSV file into a pandas DataFrame """
    return pd.read_csv(file_path)

def load_transactions(file_path: Path | None = None) -> pd.DataFrame:
    """Load the transactions dataset from a CSV file"""
    if file_path is None:
        file_path = RAW_DATA_DIR / "transactions_train.csv" # resort to defaults
        
    return _load_csv(file_path)

def load_articles(file_path: Path | None = None) -> pd.DataFrame:
    """ Load the Articles from the CSV File """
    if file_path is None:
        file_path = RAW_DATA_DIR / "articles.csv"

    return _load_csv(file_path)


def load_customers(file_path: Path | None = None) -> pd.DataFrame:
    """ Load the customer data from the CSV file """
    if file_path is None:
        file_path = RAW_DATA_DIR / "customers.csv"
    
    return _load_csv(file_path)

if __name__ == "__main__":
    transactions = load_transactions()
    print(transactions.head())