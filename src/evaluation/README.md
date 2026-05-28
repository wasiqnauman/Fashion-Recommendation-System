# Evaluation Metrics

This folder contains metrics for evaluating recommender systems.

Unlike normal classification problems, recommender systems care about the
quality of a ranked list. We usually ask: "Did the useful items appear near the
top of the recommendations?"

Each metric in `metrics.py` compares two things for one user:

```python
recommended_items = [10, 20, 30]
relevant_items = {20, 50}
```

- `recommended_items`: the ranked list returned by our recommender.
- `relevant_items`: the items the user actually interacted with in validation.

## `_top_k_items(items, k)`

Helper function used by the other metrics.

It validates that `k` is positive, then returns only the first `k` items.

Example:

```python
_top_k_items([10, 20, 30, 40], k=2)
# [10, 20]
```

## `precision_at_k`

Question:

> Of the top-k items we recommended, how many were actually relevant?

Example:

```python
recommended = [10, 20, 30]
relevant = {20, 50}
```

At `k=3`, we recommended 3 items and only `20` was relevant.

```text
precision@3 = 1 / 3 = 0.333
```

Precision rewards recommendations that do not waste slots.

## `recall_at_k`

Question:

> Of all relevant items, how many did we recover in the top-k?

Example:

```python
recommended = [10, 20, 30]
relevant = {20, 50}
```

The user had 2 relevant items: `20` and `50`. We found only `20`.

```text
recall@3 = 1 / 2 = 0.5
```

Recall rewards finding as many true future purchases as possible.

## `average_precision_at_k`

Question:

> Did we place relevant items early in the ranking?

Example:

```python
recommended = [20, 10, 50]
relevant = {20, 50}
```

Relevant hits appear at rank 1 and rank 3.

```text
precision at rank 1 = 1 / 1 = 1.0
precision at rank 3 = 2 / 3 = 0.667
AP@3 = (1.0 + 0.667) / 2 = 0.833
```

Average precision is stronger than simple recall because ordering matters.

## `dcg_at_k`

Question:

> How much useful content did we recommend, with higher-ranked hits worth more?

DCG stands for discounted cumulative gain.

For binary relevance, each relevant item gets credit based on its rank:

```text
gain = 1 / log2(rank + 1)
```

So:

- rank 1 gets `1.0`
- rank 2 gets about `0.63`
- rank 3 gets `0.5`

This means a relevant item at rank 1 is more valuable than a relevant item at
rank 10.

## `ndcg_at_k`

Question:

> How good was our ranking compared with the ideal perfect ranking?

NDCG stands for normalized discounted cumulative gain.

It computes:

```text
NDCG = actual DCG / ideal DCG
```

The score is usually between `0` and `1`.

- `1.0` means perfect ordering.
- `0.0` means no useful items were found.

## Quick Summary

- `precision@k`: how clean are the recommendations?
- `recall@k`: how many true items did we find?
- `average_precision@k`: did we find true items early?
- `dcg@k`: how much ranked usefulness did we produce?
- `ndcg@k`: how close was our ranking to perfect?

These metrics are central to recommender systems because ranking quality matters
more than normal classification accuracy.
