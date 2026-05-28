"""Smoke-test pipeline for building ranker training data."""

import pandas as pd

from src.data.load_data import load_customers, load_transactions
from src.data.preprocess_data import preprocess_customers, preprocess_transactions
from src.data.split_data import temporal_train_validation_split
from src.features.build_features import (
    build_item_features,
    build_user_features,
    build_user_item_features,
)
from src.features.candidates import generate_popular_candidates, label_candidates


def prepare_training_data_sample(
    customer_sample_size: int = 1_000,
    candidate_top_k: int = 10,
) -> pd.DataFrame:
    """Build a small labeled feature table for pipeline validation."""
    transactions = preprocess_transactions(load_transactions())
    customers = preprocess_customers(load_customers())

    train_df, validation_df = temporal_train_validation_split(transactions)

    sampled_customers = customers[["customer_id"]].head(customer_sample_size)
    user_features = build_user_features(train_df)
    item_features = build_item_features(train_df)

    candidates = generate_popular_candidates(
        train_transactions=train_df,
        customers=sampled_customers,
        top_k=candidate_top_k,
    )
    labeled_candidates = label_candidates(candidates, validation_df)

    training_features = build_user_item_features(
        candidates=labeled_candidates,
        user_features=user_features,
        item_features=item_features,
    )

    return training_features


if __name__ == "__main__":
    features = prepare_training_data_sample()
    print("Training feature sample:", features.shape)
    print(features.head())
    print("Purchased label counts:")
    print(features["purchased"].value_counts())
