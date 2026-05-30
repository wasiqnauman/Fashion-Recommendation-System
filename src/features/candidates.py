"""Candidate generation utilities for recommendation models."""

import pandas as pd


def generate_popular_candidates(
    train_transactions: pd.DataFrame,
    customers: pd.DataFrame,
    top_k: int = 100,
) -> pd.DataFrame:
    """Generate candidate items for each customer using global popularity."""
    top_articles = (
        train_transactions.groupby("article_id")
        .size()
        .reset_index(name="purchase_count")
        .sort_values("purchase_count", ascending=False)
        .head(top_k)
    )

    customer_ids = customers[["customer_id"]].drop_duplicates()
    candidates = customer_ids.merge(top_articles[["article_id"]], how="cross")
    candidates["candidate_source"] = "popular"

    return candidates


def generate_recent_popular_candidates(
    train_transactions: pd.DataFrame,
    customers: pd.DataFrame,
    days: int = 7,
    top_k: int = 100,
) -> pd.DataFrame:
    """Generate candidate items from the most popular recent purchases."""
    latest_date = train_transactions["t_dat"].max()
    cutoff_date = latest_date - pd.Timedelta(days=days)
    recent_transactions = train_transactions[
        train_transactions["t_dat"] >= cutoff_date
    ]

    top_articles = (
        recent_transactions.groupby("article_id")
        .size()
        .reset_index(name="purchase_count")
        .sort_values("purchase_count", ascending=False)
        .head(top_k)
    )

    customer_ids = customers[["customer_id"]].drop_duplicates()
    candidates = customer_ids.merge(top_articles[["article_id"]], how="cross")
    candidates["candidate_source"] = "recent_popular"

    return candidates


def generate_user_history_candidates(
    train_transactions: pd.DataFrame,
    top_k: int = 50,
) -> pd.DataFrame:
    """Generate candidates from each customer's own recent purchase history."""
    candidates = (
        train_transactions.sort_values(
            ["customer_id", "t_dat"],
            ascending=[True, False],
        )
        .drop_duplicates(["customer_id", "article_id"])
        .groupby("customer_id")
        .head(top_k)
        [["customer_id", "article_id"]]
        .reset_index(drop=True)
    )
    candidates["candidate_source"] = "user_history"

    return candidates


def merge_candidate_sources(candidate_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge candidate DataFrames and remove duplicate customer-item pairs."""
    if not candidate_frames:
        raise ValueError("candidate_frames must contain at least one DataFrame.")

    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates = candidates.drop_duplicates(
        subset=["customer_id", "article_id"]
    ).reset_index(drop=True)

    return candidates


def label_candidates(
    candidates: pd.DataFrame, validation_transactions: pd.DataFrame
) -> pd.DataFrame:
    """Add a binary label showing whether each candidate was purchased."""
    validation_pairs = validation_transactions[
        ["customer_id", "article_id"]
    ].drop_duplicates()
    validation_pairs["purchased"] = 1

    labeled_candidates = candidates.merge(
        validation_pairs,
        on=["customer_id", "article_id"],
        how="left",
    )
    labeled_candidates["purchased"] = (
        labeled_candidates["purchased"].fillna(0).astype(int)
    )

    return labeled_candidates
