"""Utilities for validating raw recommendation datasets."""

import pandas as pd

REQUIRED_TRANSACTION_COLUMNS: set[str] = {"customer_id", "article_id", "t_dat"}
REQUIRED_ARTICLE_COLUMNS: set[str] = {"article_id"}
REQUIRED_CUSTOMER_COLUMNS: set[str] = {"customer_id"}


def _check_required_columns(
    df: pd.DataFrame, required_columns: set[str], dataset_name: str
) -> None:
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Missing required columns in {dataset_name}: {missing_list}"
        )


def validate_transactions(df: pd.DataFrame) -> None:
    _check_required_columns(df, REQUIRED_TRANSACTION_COLUMNS, "transactions")

def validate_articles(df: pd.DataFrame) -> None:
    _check_required_columns(df, REQUIRED_ARTICLE_COLUMNS, "articles")

def validate_customers(df: pd.DataFrame) -> None:
    _check_required_columns(df, REQUIRED_CUSTOMER_COLUMNS, "customers")


    
