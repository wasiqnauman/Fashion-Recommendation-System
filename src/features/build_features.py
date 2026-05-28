"""Feature engineering utilities for recommendation models."""

import pandas as pd


def build_user_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Create basic user-level interaction features. (interaction = purchase)"""
    latest_date = transactions["t_dat"].max()

    user_features = (
        transactions.groupby("customer_id")
        .agg(
            user_interaction_count=("article_id", "count"),
            user_unique_items=("article_id", "nunique"),
            user_last_purchase_date=("t_dat", "max"),
        )
        .reset_index()
    )
    user_features["user_days_since_last_purchase"] = (
        latest_date - user_features["user_last_purchase_date"]
    ).dt.days
    return user_features


def build_item_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Create basic item-level interaction features."""
    latest_date = transactions["t_dat"].max()

    item_features = (
        transactions.groupby("article_id")
        .agg(
            item_interaction_count=("customer_id", "count"),
            item_unique_users=("customer_id", "nunique"),
            item_last_purchase_date=("t_dat", "max"),
        )
        .reset_index()
    )
    item_features["item_days_since_last_purchase"] = (
        latest_date - item_features["item_last_purchase_date"]
    ).dt.days
    return item_features


def build_popularity_features(
    transactions: pd.DataFrame, top_k: int = 100
) -> pd.DataFrame:
    """Create a simple popularity baseline from transaction counts."""
    popularity_features = (
        transactions.groupby("article_id")
        .size()
        .reset_index(name="purchase_count")
        .sort_values("purchase_count", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    popularity_features["popularity_rank"] = (
        popularity_features.index + 1
    )
    return popularity_features


def build_user_item_features(
    candidates: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach user and item features to candidate customer-item pairs."""
    candidate_features = candidates.merge(
        user_features,
        on="customer_id",
        how="left",
    )
    candidate_features = candidate_features.merge(
        item_features,
        on="article_id",
        how="left",
    )

    numeric_columns = candidate_features.select_dtypes(include="number").columns
    candidate_features[numeric_columns] = candidate_features[numeric_columns].fillna(0)

    return candidate_features

