"""Utilities for preprocessing recommendation datasets."""

import pandas as pd


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Convert transaction dates and sort interactions chronologically."""
    processed_df = df.copy()
    processed_df["t_dat"] = pd.to_datetime(processed_df["t_dat"])
    processed_df = processed_df.drop_duplicates()
    processed_df = processed_df.sort_values("t_dat").reset_index(drop=True)
    return processed_df


def _fill_unknown_values(
    df: pd.DataFrame, columns: list[str], fill_value: str = "Unknown" ) -> pd.DataFrame:
    """Fill missing values in selected categorical columns."""
    processed_df = df.copy()
    available_columns = [column for column in columns if column in processed_df.columns]
    if available_columns:
        processed_df[available_columns] = processed_df[available_columns].fillna(
            fill_value
        )
    return processed_df


def preprocess_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Apply light cleaning to article metadata."""
    processed_df = df.copy()
    categorical_columns = [
        "prod_name",
        "product_type_name",
        "product_group_name",
        "graphical_appearance_name",
        "colour_group_name",
        "perceived_colour_value_name",
        "perceived_colour_master_name",
        "department_name",
        "index_name",
        "index_group_name",
        "section_name",
        "garment_group_name",
        "detail_desc",
    ]
    processed_df = _fill_unknown_values(processed_df, categorical_columns)
    processed_df = processed_df.drop_duplicates(subset=["article_id"]).reset_index(
        drop=True
    )
    return processed_df


def preprocess_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Apply light cleaning to customer metadata."""
    processed_df = df.copy()
    categorical_columns = [
        "FN",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
        "postal_code",
    ]
    processed_df = _fill_unknown_values(processed_df, categorical_columns)
    if "age" in processed_df.columns:
        processed_df["age"] = processed_df["age"].fillna(processed_df["age"].median())
    processed_df = processed_df.drop_duplicates(subset=["customer_id"]).reset_index(
        drop=True
    )
    return processed_df
