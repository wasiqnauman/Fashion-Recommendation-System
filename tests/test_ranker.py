import pandas as pd
import pytest

from src.models.ranker import (
    get_numeric_feature_columns,
    rank_recommendations,
    train_logistic_ranker,
)


def test_get_numeric_feature_columns_excludes_ids_labels_and_dates():
    training_features = pd.DataFrame(
        {
            "customer_id": ["c1"],
            "article_id": [101],
            "candidate_source": ["popular"],
            "purchased": [1],
            "user_last_purchase_date": [pd.Timestamp("2020-01-01")],
            "item_last_purchase_date": [pd.Timestamp("2020-01-01")],
            "user_interaction_count": [3],
            "item_interaction_count": [5],
        }
    )

    columns = get_numeric_feature_columns(training_features)

    assert columns == ["user_interaction_count", "item_interaction_count"]


def test_train_logistic_ranker_requires_positive_and_negative_examples():
    training_features = pd.DataFrame(
        {
            "user_interaction_count": [1, 2],
            "item_interaction_count": [3, 4],
            "purchased": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="positive and negative"):
        train_logistic_ranker(
            training_features,
            feature_columns=["user_interaction_count", "item_interaction_count"],
        )


def test_rank_recommendations_sorts_per_customer_and_keeps_top_k():
    scored_candidates = pd.DataFrame(
        {
            "customer_id": ["c1", "c1", "c1", "c2"],
            "article_id": [103, 101, 102, 201],
            "score": [0.4, 0.9, 0.9, 0.2],
        }
    )

    recommendations = rank_recommendations(scored_candidates, top_k=2)

    assert recommendations.to_dict("records") == [
        {"customer_id": "c1", "article_id": 101, "score": 0.9, "rank": 1},
        {"customer_id": "c1", "article_id": 102, "score": 0.9, "rank": 2},
        {"customer_id": "c2", "article_id": 201, "score": 0.2, "rank": 1},
    ]


def test_rank_recommendations_requires_positive_top_k():
    scored_candidates = pd.DataFrame(
        {"customer_id": ["c1"], "article_id": [101], "score": [0.1]}
    )

    with pytest.raises(ValueError, match="positive"):
        rank_recommendations(scored_candidates, top_k=0)
