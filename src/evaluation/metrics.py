"""Top-k evaluation metrics for recommender systems."""

import math
from collections.abc import Iterable


def _top_k_items(items: Iterable, k: int) -> list:
    """Return the first k items after validating k."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    return list(items)[:k]


def precision_at_k(recommended_items: Iterable, relevant_items: set, k: int) -> float:
    """Calculate the fraction of top-k recommendations that are relevant."""
    top_k_items = _top_k_items(recommended_items, k)
    if not top_k_items:
        return 0.0

    hits = sum(item in relevant_items for item in top_k_items)
    return hits / len(top_k_items)


def recall_at_k(recommended_items: Iterable, relevant_items: set, k: int) -> float:
    """Calculate the fraction of relevant items recovered in the top k."""
    if not relevant_items:
        return 0.0

    top_k_items = _top_k_items(recommended_items, k)
    hits = sum(item in relevant_items for item in top_k_items)
    return hits / len(relevant_items)


def average_precision_at_k(
    recommended_items: Iterable, relevant_items: set, k: int
) -> float:
    """Calculate average precision over relevant hits in the top k."""
    if not relevant_items:
        return 0.0

    top_k_items = _top_k_items(recommended_items, k)
    hit_count = 0
    precision_sum = 0.0

    for rank, item in enumerate(top_k_items, start=1):
        if item in relevant_items:
            hit_count += 1
            precision_sum += hit_count / rank

    return precision_sum / min(len(relevant_items), k)


def dcg_at_k(recommended_items: Iterable, relevant_items: set, k: int) -> float:
    """Calculate discounted cumulative gain for binary relevance."""
    top_k_items = _top_k_items(recommended_items, k)
    dcg = 0.0

    for rank, item in enumerate(top_k_items, start=1):
        if item in relevant_items:
            dcg += 1 / math.log2(rank + 1)

    return dcg


def ndcg_at_k(recommended_items: Iterable, relevant_items: set, k: int) -> float:
    """Calculate normalized discounted cumulative gain in the top k."""
    if not relevant_items:
        return 0.0

    ideal_hits = min(len(relevant_items), k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(recommended_items, relevant_items, k) / ideal_dcg
