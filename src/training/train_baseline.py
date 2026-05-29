"""Train and evaluate the popularity baseline recommender."""

from src.data.load_data import load_customers, load_transactions
from src.data.preprocess_data import preprocess_customers, preprocess_transactions
from src.data.split_data import temporal_train_validation_split
from src.evaluation.evaluate import evaluate_recommendations
from src.models.popularity import fit_popularity_model, recommend_popular_items


def run_popularity_baseline(top_k: int = 12):
    """Run the popularity baseline and return evaluation metrics."""
    transactions = preprocess_transactions(load_transactions())
    customers = preprocess_customers(load_customers())

    train_df, validation_df = temporal_train_validation_split(transactions)
    popularity_table = fit_popularity_model(train_df)

    validation_customer_ids = validation_df["customer_id"].drop_duplicates()
    known_customer_ids = customers["customer_id"].drop_duplicates()
    target_customer_ids = validation_customer_ids[
        validation_customer_ids.isin(known_customer_ids)
    ]

    recommendations = recommend_popular_items(
        popularity_table=popularity_table,
        customer_ids=target_customer_ids,
        top_k=top_k,
    )
    metrics = evaluate_recommendations(
        recommendations=recommendations,
        validation_transactions=validation_df,
        k_values=[5, 10, top_k],
    )

    return metrics


if __name__ == "__main__":
    baseline_metrics = run_popularity_baseline()
    print("Popularity baseline metrics:")
    print(baseline_metrics)
