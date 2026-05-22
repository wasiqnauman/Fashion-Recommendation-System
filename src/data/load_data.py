"""Utilities for loading raw recommendation datasets."""

from pathlib import Path
import pandas as pd
from src.data.validate_data import (validate_customers, validate_articles, validate_transactions)


RAW_DATA_DIR = Path("data/raw")


def _load_data(file_path: Path) -> pd.DataFrame:
    """ Load a parquet file into a pandas DataFrame """
    return pd.read_parquet(file_path)

def load_transactions(file_path: Path | None = None) -> pd.DataFrame:
    """Load the transactions dataset from a parquet file"""
    if file_path is None:
        file_path = RAW_DATA_DIR / "transactions_train.parquet" # resort to defaults

    df = _load_data(file_path)
    validate_transactions(df)

    return df

def load_articles(file_path: Path | None = None) -> pd.DataFrame:
    """ Load the Articles from the parquet File """
    if file_path is None:
        file_path = RAW_DATA_DIR / "articles.parquet"

    df = _load_data(file_path)
    validate_articles(df)

    return df


def load_customers(file_path: Path | None = None) -> pd.DataFrame:
    """ Load the customer data from the parquet file """
    if file_path is None:
        file_path = RAW_DATA_DIR / "customers.parquet"
    
    df = _load_data(file_path)
    validate_customers(df)

    return df

if __name__ == "__main__":
    transactions = load_transactions()
    articles = load_articles()
    customers = load_customers()
    print(f'Transactions: {transactions.shape}')
    print(f'Articles: {articles.shape}')
    print(f'Customers: {customers.shape}')