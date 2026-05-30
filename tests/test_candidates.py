import pandas as pd
import pytest

from src.features.candidates import (
    generate_popular_candidates,
    generate_recent_popular_candidates,
    generate_user_history_candidates,
    label_candidates,
    merge_candidate_sources,
)


def test_generate_popular_candidates_returns_top_items_for_each_customer():
    transactions = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3", "c4"],
            "article_id": [101, 101, 102, 103],
        }
    )
    customers = pd.DataFrame({"customer_id": ["c1", "c2"]})

    candidates = generate_popular_candidates(transactions, customers, top_k=2)

    assert list(candidates.columns) == [
        "customer_id",
        "article_id",
        "candidate_source",
    ]
    assert len(candidates) == 4
    assert set(candidates["article_id"]) == {101, 102}
    assert set(candidates["customer_id"]) == {"c1", "c2"}
    assert set(candidates["candidate_source"]) == {"popular"}


def test_generate_recent_popular_candidates_uses_recent_window():
    transactions = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3", "c4"],
            "article_id": [101, 102, 102, 103],
            "t_dat": pd.to_datetime(
                ["2020-09-01", "2020-09-20", "2020-09-21", "2020-09-21"]
            ),
        }
    )
    customers = pd.DataFrame({"customer_id": ["c1", "c2"]})

    candidates = generate_recent_popular_candidates(
        transactions,
        customers,
        days=7,
        top_k=1,
    )

    assert len(candidates) == 2
    assert set(candidates["customer_id"]) == {"c1", "c2"}
    assert set(candidates["article_id"]) == {102}
    assert set(candidates["candidate_source"]) == {"recent_popular"}


def test_generate_user_history_candidates_returns_recent_unique_user_items():
    transactions = pd.DataFrame(
        {
            "customer_id": ["c1", "c1", "c1", "c2"],
            "article_id": [101, 102, 101, 201],
            "t_dat": pd.to_datetime(
                ["2020-09-01", "2020-09-03", "2020-09-05", "2020-09-02"]
            ),
        }
    )

    candidates = generate_user_history_candidates(transactions, top_k=2)

    assert candidates.to_dict("records") == [
        {"customer_id": "c1", "article_id": 101, "candidate_source": "user_history"},
        {"customer_id": "c1", "article_id": 102, "candidate_source": "user_history"},
        {"customer_id": "c2", "article_id": 201, "candidate_source": "user_history"},
    ]


def test_merge_candidate_sources_removes_duplicate_customer_item_pairs():
    first = pd.DataFrame(
        {
            "customer_id": ["c1", "c1"],
            "article_id": [101, 102],
            "candidate_source": ["popular", "popular"],
        }
    )
    second = pd.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "article_id": [101, 101],
            "candidate_source": ["recent", "recent"],
        }
    )

    merged = merge_candidate_sources([first, second])

    assert len(merged) == 3
    assert not merged.duplicated(["customer_id", "article_id"]).any()


def test_merge_candidate_sources_requires_at_least_one_frame():
    with pytest.raises(ValueError, match="at least one"):
        merge_candidate_sources([])


def test_label_candidates_marks_validation_purchases():
    candidates = pd.DataFrame(
        {
            "customer_id": ["c1", "c1", "c2"],
            "article_id": [101, 102, 101],
            "candidate_source": ["popular", "popular", "popular"],
        }
    )
    validation = pd.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "article_id": [102, 103],
        }
    )

    labeled = label_candidates(candidates, validation)

    labels = {
        (row.customer_id, row.article_id): row.purchased
        for row in labeled.itertuples(index=False)
    }
    assert labels == {("c1", 101): 0, ("c1", 102): 1, ("c2", 101): 0}
