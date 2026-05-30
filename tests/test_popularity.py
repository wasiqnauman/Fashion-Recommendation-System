import pandas as pd
import pytest

from src.models.popularity import fit_popularity_model, recommend_popular_items


def test_fit_popularity_model_sorts_articles_by_purchase_count():
    transactions = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3", "c4", "c5"],
            "article_id": [102, 101, 101, 103, 101],
        }
    )

    popularity = fit_popularity_model(transactions)

    assert popularity["article_id"].tolist() == [101, 102, 103]
    assert popularity["purchase_count"].tolist() == [3, 1, 1]
    assert popularity["rank"].tolist() == [1, 2, 3]


def test_recommend_popular_items_repeats_top_items_for_each_customer():
    popularity = pd.DataFrame(
        {
            "article_id": [101, 102, 103],
            "purchase_count": [3, 2, 1],
            "score": [3, 2, 1],
            "rank": [1, 2, 3],
        }
    )

    recommendations = recommend_popular_items(
        popularity,
        customer_ids=["c1", "c1", "c2"],
        top_k=2,
    )

    assert len(recommendations) == 4
    assert recommendations["customer_id"].tolist() == ["c1", "c1", "c2", "c2"]
    assert recommendations["article_id"].tolist() == [101, 102, 101, 102]
    assert recommendations["rank"].tolist() == [1, 2, 1, 2]


def test_recommend_popular_items_requires_positive_top_k():
    popularity = pd.DataFrame({"article_id": [101], "score": [1]})

    with pytest.raises(ValueError, match="positive"):
        recommend_popular_items(popularity, customer_ids=["c1"], top_k=0)
