"""Simple supervised ranker utilities for recommendation."""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXCLUDED_FEATURE_COLUMNS = {
    "customer_id",
    "article_id",
    "candidate_source",
    "purchased",
    "user_last_purchase_date",
    "item_last_purchase_date",
}


def get_numeric_feature_columns(training_features: pd.DataFrame) -> list[str]:
    """Select numeric model features while excluding IDs and labels."""
    numeric_columns = training_features.select_dtypes(include="number").columns
    return [
        column
        for column in numeric_columns
        if column not in EXCLUDED_FEATURE_COLUMNS
    ]


def train_logistic_ranker(
    training_features: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "purchased",
) -> Pipeline:
    """Train a logistic regression model to score candidate purchases."""
    if training_features[target_column].nunique() < 2:
        raise ValueError(
            "Ranker training requires both positive and negative examples."
        )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(training_features[feature_columns], training_features[target_column])
    return model


def predict_scores(
    model: Pipeline,
    candidate_features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Add purchase probability scores to candidate feature rows."""
    scored_candidates = candidate_features.copy()
    scored_candidates["score"] = model.predict_proba(
        scored_candidates[feature_columns]
    )[:, 1]
    return scored_candidates


def rank_recommendations(
    scored_candidates: pd.DataFrame,
    top_k: int = 12,
) -> pd.DataFrame:
    """Rank scored candidates and keep the top k items per customer."""
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    recommendations = (
        scored_candidates.sort_values(
            ["customer_id", "score", "article_id"],
            ascending=[True, False, True],
        )
        .groupby("customer_id")
        .head(top_k)
        .copy()
    )
    recommendations["rank"] = recommendations.groupby("customer_id").cumcount() + 1

    return recommendations[["customer_id", "article_id", "score", "rank"]]


def save_model(model: Pipeline, path: Path) -> None:
    """Save a trained ranker model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as model_file:
        pickle.dump(model, model_file)


def load_model(path: Path) -> Pipeline:
    """Load a trained ranker model."""
    with path.open("rb") as model_file:
        return pickle.load(model_file)
