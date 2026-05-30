import pytest

from src.evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_recall_and_average_precision_at_k():
    recommended = [101, 102, 103, 104]
    relevant = {102, 104, 105}

    assert precision_at_k(recommended, relevant, k=3) == pytest.approx(1 / 3)
    assert recall_at_k(recommended, relevant, k=3) == pytest.approx(1 / 3)
    assert average_precision_at_k(recommended, relevant, k=4) == pytest.approx(
        ((1 / 2) + (2 / 4)) / 3
    )


def test_metrics_return_zero_when_no_relevant_items():
    assert recall_at_k([101], set(), k=1) == 0.0
    assert average_precision_at_k([101], set(), k=1) == 0.0
    assert ndcg_at_k([101], set(), k=1) == 0.0


def test_metric_k_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        precision_at_k([101], {101}, k=0)
