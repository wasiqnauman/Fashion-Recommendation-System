"""Utilities for evaluating recommendation outputs."""

import pandas as pd

from src.evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def build_ground_truth(validation_transactions: pd.DataFrame) -> pd.DataFrame:
    """Group validation purchases into relevant item sets per customer (items already purchased by the customer)"""
    ground_truth = (
        validation_transactions.groupby("customer_id")["article_id"]
        .agg(set)
        .reset_index(name="relevant_items")
    )
    return ground_truth


def evaluate_recommendations(
    recommendations: pd.DataFrame,
    validation_transactions: pd.DataFrame,
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    """Evaluate recommendations against validation purchases."""
    if k_values is None:
        k_values = [5, 10, 12]

    ground_truth = build_ground_truth(validation_transactions)
    recommendation_lists = (
        recommendations.sort_values(["customer_id", "rank"])
        .groupby("customer_id")["article_id"]
        .agg(list)
        .reset_index(name="recommended_items")
    )

    evaluation_df = ground_truth.merge(
        recommendation_lists,
        on="customer_id",
        how="left",
    )
    evaluation_df["recommended_items"] = evaluation_df["recommended_items"].apply(
        lambda items: items if isinstance(items, list) else []
    )

    metric_rows = []
    for k in k_values:
        metric_rows.append(
            {
                "k": k,
                "precision_at_k": evaluation_df.apply(
                    lambda row: precision_at_k(
                        row["recommended_items"], row["relevant_items"], k
                    ),
                    axis=1,
                ).mean(),
                "recall_at_k": evaluation_df.apply(
                    lambda row: recall_at_k(
                        row["recommended_items"], row["relevant_items"], k
                    ),
                    axis=1,
                ).mean(),
                "map_at_k": evaluation_df.apply(
                    lambda row: average_precision_at_k(
                        row["recommended_items"], row["relevant_items"], k
                    ),
                    axis=1,
                ).mean(),
                "ndcg_at_k": evaluation_df.apply(
                    lambda row: ndcg_at_k(
                        row["recommended_items"], row["relevant_items"], k
                    ),
                    axis=1,
                ).mean(),
            }
        )

    return pd.DataFrame(metric_rows)
