"""Utilities for creating temporal train and validation splits."""

import pandas as pd

from src.data.load_data import load_transactions
from src.data.preprocess_data import preprocess_transactions


def temporal_train_validation_split(
    transactions: pd.DataFrame, validation_days: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split transactions into train and validation sets using time."""
    if validation_days <= 0:
        raise ValueError("validation_days must be a positive integer.")

    if transactions.empty:
        raise ValueError("transactions DataFrame is empty.")

    latest_date = transactions["t_dat"].max()
    cutoff_date = latest_date - pd.Timedelta(days=validation_days)

    train_df = transactions[transactions["t_dat"] < cutoff_date].copy()
    validation_df = transactions[transactions["t_dat"] >= cutoff_date].copy()

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)

    return train_df, validation_df


if __name__ == "__main__":
    transactions = load_transactions()
    transactions = preprocess_transactions(transactions)
    train_df, validation_df = temporal_train_validation_split(transactions)

    print("Train:", train_df.shape)
    print("Validation:", validation_df.shape)
    print("Train date range:", train_df["t_dat"].min(), train_df["t_dat"].max())
    print(
        "Validation date range:",
        validation_df["t_dat"].min(),
        validation_df["t_dat"].max(),
    )
