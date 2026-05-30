import pandas as pd

from src.training.train_ranker import _build_candidate_feature_table


def test_build_candidate_feature_table_merges_multiple_candidate_sources():
    feature_transactions = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3", "c2"],
            "article_id": [201, 101, 101, 301],
            "t_dat": pd.to_datetime(
                ["2020-09-02", "2020-09-01", "2020-09-01", "2020-09-10"]
            ),
        }
    )
    label_transactions = pd.DataFrame(
        {
            "customer_id": ["c1"],
            "article_id": [201],
        }
    )
    target_customers = pd.DataFrame({"customer_id": ["c1"]})

    candidate_features = _build_candidate_feature_table(
        feature_transactions=feature_transactions,
        label_transactions=label_transactions,
        target_customers=target_customers,
        candidate_top_k=1,
    )

    assert not candidate_features.duplicated(["customer_id", "article_id"]).any()
    assert set(candidate_features["article_id"]) == {101, 201, 301}
    assert set(candidate_features["candidate_source"]) == {
        "popular",
        "recent_popular",
        "user_history",
    }

    purchased_by_article = dict(
        zip(candidate_features["article_id"], candidate_features["purchased"])
    )
    assert purchased_by_article == {101: 0, 201: 1, 301: 0}
