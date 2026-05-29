"""Popularity baseline model for recommendation."""

import pandas as pd


def fit_popularity_model(train_transactions: pd.DataFrame) -> pd.DataFrame:
    """Fit a popularity model by counting purchases per article."""
    popularity_table = (
        train_transactions.groupby("article_id")
        .size()
        .reset_index(name="purchase_count")
        .sort_values("purchase_count", ascending=False)
        .reset_index(drop=True)
    )
    popularity_table["score"] = popularity_table["purchase_count"]
    popularity_table["rank"] = popularity_table.index + 1
    return popularity_table


def recommend_popular_items(
    popularity_table: pd.DataFrame,
    customer_ids: pd.Series | list[str],
    top_k: int = 12,
) -> pd.DataFrame:
    """Recommend the same top popular items to each customer."""
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    top_items = popularity_table[["article_id", "score"]].head(top_k).copy()
    top_items["rank"] = range(1, len(top_items) + 1)

    customers = pd.DataFrame({"customer_id": pd.Series(customer_ids).drop_duplicates()})
    recommendations = customers.merge(top_items, how="cross")

    return recommendations[["customer_id", "article_id", "score", "rank"]]
