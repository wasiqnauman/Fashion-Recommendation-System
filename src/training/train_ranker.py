"""Train and evaluate a simple supervised recommendation ranker."""

import pandas as pd

from src.data.load_data import load_customers, load_transactions
from src.data.preprocess_data import preprocess_customers, preprocess_transactions
from src.data.split_data import temporal_train_validation_split
from src.evaluation.evaluate import evaluate_recommendations
from src.features.build_features import (
    build_item_features,
    build_user_features,
    build_user_item_features,
)
from src.features.candidates import generate_popular_candidates, label_candidates
from src.models.ranker import (
    get_numeric_feature_columns,
    predict_scores,
    rank_recommendations,
    train_logistic_ranker,
)


def _sample_customer_ids(
    transactions: pd.DataFrame,
    customer_sample_size: int,
) -> pd.DataFrame:
    """Select a deterministic customer sample from a transaction window."""
    customer_ids = transactions["customer_id"].drop_duplicates().head(
        customer_sample_size
    )
    return pd.DataFrame({"customer_id": customer_ids})


def _build_candidate_feature_table(
    feature_transactions: pd.DataFrame,
    label_transactions: pd.DataFrame,
    target_customers: pd.DataFrame,
    candidate_top_k: int,
) -> pd.DataFrame:
    """Generate, label, and featurize candidates for one time window."""
    user_features = build_user_features(feature_transactions)
    item_features = build_item_features(feature_transactions)

    candidates = generate_popular_candidates(
        train_transactions=feature_transactions,
        customers=target_customers,
        top_k=candidate_top_k,
    )
    labeled_candidates = label_candidates(candidates, label_transactions)

    return build_user_item_features(
        candidates=labeled_candidates,
        user_features=user_features,
        item_features=item_features,
    )


def run_logistic_ranker_experiment(
    customer_sample_size: int = 5_000,
    candidate_top_k: int = 50,
    recommendation_top_k: int = 12,
) -> pd.DataFrame:
    """Train a logistic ranker on one time window and evaluate on the next."""
    transactions = preprocess_transactions(load_transactions())
    customers = preprocess_customers(load_customers())

    train_and_label_df, evaluation_df = temporal_train_validation_split(
        transactions,
        validation_days=7,
    )
    feature_train_df, ranker_label_df = temporal_train_validation_split(
        train_and_label_df,
        validation_days=7,
    )

    train_customers = _sample_customer_ids(
        ranker_label_df,
        customer_sample_size=customer_sample_size,
    )
    train_features = _build_candidate_feature_table(
        feature_transactions=feature_train_df,
        label_transactions=ranker_label_df,
        target_customers=train_customers,
        candidate_top_k=candidate_top_k,
    )

    feature_columns = get_numeric_feature_columns(train_features)
    model = train_logistic_ranker(train_features, feature_columns)

    evaluation_customers = _sample_customer_ids(
        evaluation_df,
        customer_sample_size=customer_sample_size,
    )
    known_customers = customers[["customer_id"]].drop_duplicates()
    evaluation_customers = evaluation_customers.merge(
        known_customers,
        on="customer_id",
        how="inner",
    )

    evaluation_features = _build_candidate_feature_table(
        feature_transactions=train_and_label_df,
        label_transactions=evaluation_df,
        target_customers=evaluation_customers,
        candidate_top_k=candidate_top_k,
    )
    scored_candidates = predict_scores(model, evaluation_features, feature_columns)
    recommendations = rank_recommendations(scored_candidates, top_k=recommendation_top_k)

    metrics = evaluate_recommendations(
        recommendations=recommendations,
        validation_transactions=evaluation_df[
            evaluation_df["customer_id"].isin(evaluation_customers["customer_id"])
        ],
        k_values=[5, 10, recommendation_top_k],
    )

    print("Training features:", train_features.shape)
    print("Training label counts:")
    print(train_features["purchased"].value_counts())
    print("Evaluation features:", evaluation_features.shape)

    return metrics


if __name__ == "__main__":
    ranker_metrics = run_logistic_ranker_experiment()
    print("Logistic ranker metrics:")
    print(ranker_metrics)
