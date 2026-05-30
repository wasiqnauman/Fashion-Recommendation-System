# Fashion Recommender System Project Design

## 1. Project Goal

Build a production-style personalized fashion recommendation system using the H&M Personalized Fashion Recommendations dataset.

The final system should:

- Load and validate raw parquet datasets.
- Preprocess customer, article, and transaction data.
- Create time-based train, validation, and eventually test splits.
- Build user, item, popularity, and candidate-level features.
- Train baseline and ranking recommendation models.
- Evaluate recommendations with top-k recommender metrics.
- Serve recommendations through an API.
- Package the project with tests, configuration, documentation, and production-friendly entrypoints.

The core architecture will follow a two-stage recommender design:

1. Candidate generation: quickly produce a smaller set of likely items for each user.
2. Ranking: score and order those candidate items using engineered features and a supervised model.

This design is intentional because large-scale recommendation systems usually cannot score every item for every user. They first retrieve a manageable candidate set, then rank it with richer features.

## 2. Current Project Structure

```text
fashion-recommender-system/
  data/
    raw/
      articles.parquet
      customers.parquet
      sample_submission.parquet
      transactions_train.parquet
  src/
    api/
      __init__.py
    data/
      __init__.py
      load_data.py
      preprocess_data.py
      split_data.py
      validate_data.py
    evaluation/
      __init__.py
      evaluate.py
      metrics.py
    features/
      __init__.py
      build_features.py
      candidates.py
    inference/
      __init__.py
    models/
      __init__.py
      popularity.py
      ranker.py
    training/
      __init__.py
      prepare_training_data.py
      train_baseline.py
      train_ranker.py
  tests/
    conftest.py
    test_candidates.py
    test_metrics.py
    test_popularity.py
    test_ranker.py
    utils/
      __init__.py
    __init__.py
  README.md
  requirements.txt
  project.md
```

The project is currently in the baseline modeling and ranker experimentation stage.

## 3. Implemented Components

### 3.1 `src/data/validate_data.py`

Purpose:

Validate that raw datasets contain the minimum columns required by the pipeline.

Implemented constants:

- `REQUIRED_TRANSACTION_COLUMNS: set[str]`
  - Required columns: `customer_id`, `article_id`, `t_dat`.
  - These are the minimum fields needed to know who interacted with what and when.

- `REQUIRED_ARTICLE_COLUMNS: set[str]`
  - Required columns: `article_id`.
  - This is the minimum field needed to join item metadata to transactions.

- `REQUIRED_CUSTOMER_COLUMNS: set[str]`
  - Required columns: `customer_id`.
  - This is the minimum field needed to join customer metadata to transactions.

Implemented functions:

- `_check_required_columns(df: pd.DataFrame, required_columns: set[str], dataset_name: str) -> None`
  - Private helper used by all dataset validators.
  - Compares the required columns against the DataFrame columns.
  - Raises a `ValueError` if any required columns are missing.
  - Includes the dataset name and missing columns in the error message.
  - Design reasoning: one shared helper avoids duplicating validation logic and keeps public validators simple.

- `validate_transactions(df: pd.DataFrame) -> None`
  - Validates the transactions table.
  - Confirms that each transaction can identify the customer, item, and date.

- `validate_articles(df: pd.DataFrame) -> None`
  - Validates the articles table.
  - Confirms that article metadata can be joined by `article_id`.

- `validate_customers(df: pd.DataFrame) -> None`
  - Validates the customers table.
  - Confirms that customer metadata can be joined by `customer_id`.

Planned improvements:

- `validate_non_empty(df: pd.DataFrame, dataset_name: str) -> None`
  - Raise an error if a dataset is empty.

- `validate_no_null_ids(df: pd.DataFrame, id_columns: list[str], dataset_name: str) -> None`
  - Ensure key ID fields are not null.

- `validate_transaction_dates(df: pd.DataFrame) -> None`
  - Ensure `t_dat` exists and can be converted to datetime.

### 3.2 `src/data/load_data.py`

Purpose:

Load raw parquet datasets and immediately validate them.

Implemented constant:

- `RAW_DATA_DIR = Path("data/raw")`
  - Default location for raw dataset files.
  - Design reasoning: centralizing the raw path prevents hardcoded paths across the project.

Implemented functions:

- `_load_data(file_path: Path) -> pd.DataFrame`
  - Private helper that loads a parquet file using `pd.read_parquet`.
  - Current name is generic.
  - Recommended future rename: `_load_parquet_table`.

- `load_transactions(file_path: Path | None = None) -> pd.DataFrame`
  - Loads `transactions_train.parquet` by default.
  - Calls `validate_transactions`.
  - Returns a validated transactions DataFrame.
  - Design reasoning: callers should receive data that has passed basic schema checks.

- `load_articles(file_path: Path | None = None) -> pd.DataFrame`
  - Loads `articles.parquet` by default.
  - Calls `validate_articles`.
  - Returns a validated articles DataFrame.

- `load_customers(file_path: Path | None = None) -> pd.DataFrame`
  - Loads `customers.parquet` by default.
  - Calls `validate_customers`.
  - Returns a validated customers DataFrame.

Implemented runnable behavior:

- Running `python -m src.data.load_data` loads all three datasets and prints their shapes.

Planned improvements:

- Add file existence checks before reading parquet files.
- Improve formatting and import style.
- Move path values into `configs/config.yaml`.
- Keep loading and validation together, but keep preprocessing separate.

Design reasoning:

Loading should answer: "Can we read this dataset, and does it have the minimum structure required?" It should not clean or transform business data. That keeps the loading layer predictable.

### 3.3 `src/data/preprocess_data.py`

Purpose:

Convert raw validated datasets into cleaner, analysis-ready DataFrames.

Implemented functions:

- `preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame`
  - Copies the input DataFrame.
  - Converts `t_dat` to pandas datetime.
  - Drops exact duplicate transaction rows.
  - Sorts transactions chronologically by `t_dat`.
  - Resets the index.
  - Design reasoning: recommender evaluation depends on time ordering, so date parsing and sorting happen early.

- `_fill_unknown_values(df: pd.DataFrame, columns: list[str], fill_value: str = "Unknown") -> pd.DataFrame`
  - Private helper for categorical missing-value handling.
  - Copies the input DataFrame.
  - Finds which requested columns actually exist.
  - Fills missing values in available columns with `"Unknown"`.
  - Design reasoning: some metadata columns are useful but not strictly required. This helper keeps preprocessing flexible.

- `preprocess_articles(df: pd.DataFrame) -> pd.DataFrame`
  - Copies article metadata.
  - Fills missing values in selected categorical/text columns.
  - Drops duplicate rows by `article_id`.
  - Resets the index.
  - Design reasoning: article metadata will later become ranking features, so missing categorical values should be explicit rather than null.

- `preprocess_customers(df: pd.DataFrame) -> pd.DataFrame`
  - Copies customer metadata.
  - Fills missing values in selected categorical fields.
  - Fills missing `age` values with median age if the column exists.
  - Drops duplicate rows by `customer_id`.
  - Resets the index.
  - Design reasoning: categorical fields can use `"Unknown"`, but numeric fields like `age` need numeric imputation.

Planned improvements:

- `preprocess_all(transactions: pd.DataFrame, articles: pd.DataFrame, customers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
  - Runs all preprocessors in one function.

- `normalize_article_ids(df: pd.DataFrame) -> pd.DataFrame`
  - Ensure `article_id` has consistent type.

- `normalize_customer_ids(df: pd.DataFrame) -> pd.DataFrame`
  - Ensure `customer_id` has consistent type.

### 3.4 `src/data/split_data.py`

Purpose:

Create time-aware training and validation splits.

Implemented functions:

- `temporal_train_validation_split(transactions: pd.DataFrame, validation_days: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]`
  - Validates that `validation_days` is positive.
  - Validates that transactions are not empty.
  - Finds the latest transaction date.
  - Computes a cutoff date by subtracting `validation_days`.
  - Places rows before the cutoff in train.
  - Places rows on or after the cutoff in validation.
  - Returns copied and re-indexed train and validation DataFrames.

Implemented runnable behavior:

- Running `python -m src.data.split_data`:
  - loads transactions
  - preprocesses them
  - splits them
  - prints train and validation shapes
  - prints train and validation date ranges

Observed current split:

```text
Train: (28571904, 5)
Validation: (241515, 5)
Train date range: 2018-09-20 to 2020-09-14
Validation date range: 2020-09-15 to 2020-09-22
```

Design reasoning:

Recommendation models should be evaluated on future behavior. A random split would leak future information into training and make evaluation unrealistically optimistic.

Planned improvements:

- `temporal_train_validation_test_split(transactions: pd.DataFrame, validation_days: int = 7, test_days: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
  - Create train, validation, and test windows.
  - Validation is for model selection.
  - Test is for final reporting.

- `get_date_range(df: pd.DataFrame, date_column: str = "t_dat") -> tuple[pd.Timestamp, pd.Timestamp]`
  - Utility for reporting date ranges.

### 3.5 `src/features/build_features.py`

Purpose:

Build reusable feature tables from transaction history.

Implemented functions:

- `build_user_features(transactions: pd.DataFrame) -> pd.DataFrame`
  - Groups transactions by `customer_id`.
  - Creates:
    - `user_interaction_count`: total purchases by user.
    - `user_unique_items`: number of unique articles purchased.
    - `user_last_purchase_date`: most recent purchase date.
    - `user_days_since_last_purchase`: recency feature relative to the latest transaction date.
  - Design reasoning: user activity, diversity, and recency are strong baseline signals.

- `build_item_features(transactions: pd.DataFrame) -> pd.DataFrame`
  - Groups transactions by `article_id`.
  - Creates:
    - `item_interaction_count`: total item purchases.
    - `item_unique_users`: number of unique customers who purchased the item.
    - `item_last_purchase_date`: most recent item purchase date.
    - `item_days_since_last_purchase`: item recency feature.
  - Design reasoning: popularity and recent activity are strong item-level ranking signals.

- `build_popularity_features(transactions: pd.DataFrame, top_k: int = 100) -> pd.DataFrame`
  - Counts purchases per item.
  - Sorts items by purchase count.
  - Keeps the top `k` items.
  - Adds `popularity_rank`.
  - Design reasoning: popularity is the simplest recommender baseline and must be beaten by more advanced models.

- `build_user_item_features(candidates: pd.DataFrame, user_features: pd.DataFrame, item_features: pd.DataFrame) -> pd.DataFrame`
  - Join candidate pairs to user and item feature tables.
  - Output one row per `(customer_id, article_id)` candidate.
  - Fill missing numeric features with `0`.

Planned improvements:

- `add_article_metadata_features(candidate_features: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame`
  - Join article metadata onto candidate feature rows.

- `build_training_features(train_transactions: pd.DataFrame, candidate_pairs: pd.DataFrame, articles: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame`
  - Build the full supervised ranking feature table.

## 4. Component Status And Remaining Work

### 4.1 `src/data/make_dataset.py`

Purpose:

Create processed datasets from raw parquet files.

Planned functions:

- `build_processed_dataset(config_path: str | Path = "configs/config.yaml") -> None`
  - Load raw transactions, articles, and customers.
  - Validate and preprocess all datasets.
  - Create train, validation, and test splits.
  - Save processed parquet outputs.

- `save_dataframe(df: pd.DataFrame, path: Path) -> None`
  - Create parent directories if needed.
  - Save a DataFrame as parquet.

- `load_processed_splits(processed_dir: Path) -> dict[str, pd.DataFrame]`
  - Load processed train, validation, test, articles, and customers files.

Expected outputs:

```text
data/processed/transactions_train.parquet
data/processed/transactions_validation.parquet
data/processed/transactions_test.parquet
data/processed/articles.parquet
data/processed/customers.parquet
```

Design reasoning:

Training code should consume processed data. It should not repeat raw loading and preprocessing logic.

### 4.2 `src/features/candidates.py`

Purpose:

Generate candidate `(customer_id, article_id)` pairs before ranking.

Implemented functions:

- `generate_popular_candidates(train_transactions: pd.DataFrame, customers: pd.DataFrame, top_k: int = 100) -> pd.DataFrame`
  - Find the top `k` most purchased articles.
  - Assign those articles to every target customer.
  - Return columns: `customer_id`, `article_id`, `candidate_source`.

- `generate_recent_popular_candidates(train_transactions: pd.DataFrame, customers: pd.DataFrame, days: int = 7, top_k: int = 100) -> pd.DataFrame`
  - Filter transactions to the most recent `days`.
  - Find the top recent items.
  - Assign them to each customer.

- `generate_user_history_candidates(train_transactions: pd.DataFrame, top_k: int = 50) -> pd.DataFrame`
  - Generate candidates from each customer's own recent purchase history.
  - Keep each customer's most recent unique articles.

- `merge_candidate_sources(candidate_frames: list[pd.DataFrame]) -> pd.DataFrame`
  - Concatenate multiple candidate sources.
  - Drop duplicate `(customer_id, article_id)` pairs.

- `label_candidates(candidates: pd.DataFrame, validation_transactions: pd.DataFrame) -> pd.DataFrame`
  - Add binary target column `purchased`.
  - `purchased = 1` if the customer bought the candidate article during validation.
  - `purchased = 0` otherwise.

Design reasoning:

Candidate generation and ranking must stay separate. Candidate generation controls what can be recommended; ranking controls ordering.

### 4.3 `src/features/cooccurrence.py`

Purpose:

Build item-item collaborative filtering features and candidates.

Planned functions:

- `build_item_cooccurrence(transactions: pd.DataFrame, max_items_per_user: int = 50) -> pd.DataFrame`
  - For each user, collect items they bought.
  - Generate item pairs that co-occur in the same user's history.
  - Count co-occurrence frequency.
  - Return item similarity table.

- `generate_cooccurrence_candidates(train_transactions: pd.DataFrame, similarity_table: pd.DataFrame, top_k: int = 50) -> pd.DataFrame`
  - For each user, find items similar to their purchased items.
  - Return candidate pairs.

Design reasoning:

Co-occurrence is a practical collaborative filtering baseline that does not require neural models.

### 4.4 `src/models/popularity.py`

Purpose:

Implement a non-personalized popularity recommender baseline.

Implemented functions:

- `fit_popularity_model(train_transactions: pd.DataFrame) -> pd.DataFrame`
  - Count purchases by `article_id`.
  - Return a sorted item popularity table with `purchase_count`, `score`, and `rank`.

- `recommend_popular_items(popularity_table: pd.DataFrame, customer_ids: pd.Series | list[str], top_k: int = 12) -> pd.DataFrame`
  - Return top popular items for each customer.
  - Columns: `customer_id`, `article_id`, `score`, `rank`.
  - Validate that `top_k` is positive.

Design reasoning:

The popularity model is the baseline. More complex systems should be judged by whether they beat it.

### 4.5 `src/models/cooccurrence.py`

Purpose:

Implement a personalized item similarity recommender.

Planned functions:

- `fit_cooccurrence_model(train_transactions: pd.DataFrame) -> pd.DataFrame`
  - Build item-item similarity from user purchase histories.

- `recommend_similar_items(user_history: pd.DataFrame, similarity_table: pd.DataFrame, top_k: int = 12) -> pd.DataFrame`
  - Recommend items similar to each user's past purchases.

Design reasoning:

This gives the project a personalized baseline before training a supervised ranker.

### 4.6 `src/models/ranker.py`

Purpose:

Train and use a supervised ranking model.

Implemented functions:

- `get_numeric_feature_columns(training_features: pd.DataFrame) -> list[str]`
  - Select numeric feature columns.
  - Exclude IDs, labels, candidate source, and date columns.

- `train_logistic_ranker(training_features: pd.DataFrame, feature_columns: list[str], target_column: str = "purchased") -> Pipeline`
  - Train a scikit-learn logistic regression pipeline to score candidate purchases.
  - Use balanced class weights because purchased candidates are sparse.
  - Raise a `ValueError` if the training labels do not contain both positive and negative examples.

- `predict_scores(model, candidate_features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame`
  - Add purchase probability scores to candidate rows.

- `rank_recommendations(scored_candidates: pd.DataFrame, top_k: int = 12) -> pd.DataFrame`
  - Sort candidates by customer and score.
  - Return top `k` items per customer.

- `save_model(model, path: Path) -> None`
  - Save model artifact using `pickle`.

- `load_model(path: Path)`
  - Load saved model artifact.

Design reasoning:

Use logistic regression for the first supervised ranker because it is already available through scikit-learn, fast to train on samples, and simple to debug. LightGBM remains a later upgrade once candidate generation and artifact handling are stronger.

### 4.7 `src/training/train_baseline.py`

Purpose:

Train and evaluate baseline recommenders.

Implemented functions:

- `run_popularity_baseline(top_k: int = 12)`
  - Load and preprocess raw transactions and customers.
  - Create a temporal train-validation split.
  - Fit the popularity model.
  - Recommend top popular items to validation customers.
  - Evaluate recommendations with top-k metrics.

Implemented runnable behavior:

- Running `python -m src.training.train_baseline` prints popularity baseline metrics.

Planned improvements:

- Save the popularity artifact to `artifacts/models/popularity.parquet`.
- Save evaluation metrics to `artifacts/reports/`.

Expected output:

```text
artifacts/models/popularity.parquet
```

### 4.8 `src/training/prepare_training_data.py`

Purpose:

Smoke-test the labeled ranker feature pipeline.

Implemented functions:

- `prepare_training_data_sample(customer_sample_size: int = 1_000, candidate_top_k: int = 10) -> pd.DataFrame`
  - Load and preprocess transactions and customers.
  - Create a temporal train-validation split.
  - Generate popular candidates for a customer sample.
  - Label candidates from validation purchases.
  - Attach user and item features.

Implemented runnable behavior:

- Running `python -m src.training.prepare_training_data` prints the sample feature shape, head, and label counts.

### 4.9 `src/training/train_ranker.py`

Purpose:

Train the supervised ranking model.

Implemented functions:

- `_sample_customer_ids(transactions: pd.DataFrame, customer_sample_size: int) -> pd.DataFrame`
  - Select a deterministic customer sample from a transaction window.

- `_build_candidate_feature_table(feature_transactions: pd.DataFrame, label_transactions: pd.DataFrame, target_customers: pd.DataFrame, candidate_top_k: int) -> pd.DataFrame`
  - Build user features, item features, popular candidates, labels, and joined candidate features for one time window.

- `run_logistic_ranker_experiment(customer_sample_size: int = 5_000, candidate_top_k: int = 50, recommendation_top_k: int = 12) -> pd.DataFrame`
  - Train a logistic ranker on one time window.
  - Evaluate recommendations on the next time window.
  - Return top-k metric results.

Implemented runnable behavior:

- Running `python -m src.training.train_ranker` prints training feature shape, label counts, evaluation feature shape, and ranker metrics.

Planned improvements:

- Save the trained model, feature columns, training summary, and evaluation report.

Expected outputs:

```text
artifacts/models/ranker.joblib
artifacts/features/feature_columns.json
artifacts/reports/training_summary.json
```

### 4.10 `src/evaluation/metrics.py`

Purpose:

Implement recommender-specific evaluation metrics.

Implemented functions:

- `precision_at_k(recommended_items: list, relevant_items: set, k: int) -> float`
  - Fraction of top-k recommended items that are relevant.

- `recall_at_k(recommended_items: list, relevant_items: set, k: int) -> float`
  - Fraction of relevant items recovered in top-k.

- `average_precision_at_k(recommended_items: list, relevant_items: set, k: int) -> float`
  - Precision averaged at each relevant hit.

- `dcg_at_k(recommended_items: list, relevant_items: set, k: int) -> float`
  - Discounted cumulative gain.

- `ndcg_at_k(recommended_items: list, relevant_items: set, k: int) -> float`
  - Normalized ranking quality.

Design reasoning:

Classification accuracy is not appropriate for recommender systems. Top-k metrics measure whether useful items appear near the top.

### 4.11 `src/evaluation/evaluate.py`

Purpose:

Evaluate recommendation outputs against validation or test purchases.

Implemented functions:

- `build_ground_truth(transactions: pd.DataFrame) -> pd.DataFrame`
  - Group purchased `article_id` values by `customer_id`.

- `evaluate_recommendations(recommendations: pd.DataFrame, ground_truth_transactions: pd.DataFrame, k_values: list[int] = [5, 10, 12]) -> pd.DataFrame`
  - Compute recommender metrics for each `k`.
  - Return `precision_at_k`, `recall_at_k`, `map_at_k`, and `ndcg_at_k` rows.

Planned improvements:

- `save_evaluation_report(metrics_df: pd.DataFrame, path: Path) -> None`
  - Save evaluation results.

### 4.12 `src/training/evaluate_model.py`

Purpose:

Run model evaluation from saved artifacts.

Planned functions:

- `generate_validation_recommendations(config: dict) -> pd.DataFrame`
  - Load ranker and processed validation data.
  - Generate recommendations.

- `evaluate_saved_model(config_path: str | Path = "configs/config.yaml") -> None`
  - Generate recommendations.
  - Evaluate them.
  - Save report.

- `main() -> None`
  - Command-line entrypoint.

### 4.13 `src/inference/recommender.py`

Purpose:

Provide reusable recommendation logic for API and batch inference.

Planned functions:

- `load_recommender_artifacts(config: dict) -> dict`
  - Load ranker, feature columns, popularity fallback, article metadata, and processed data.

- `recommend_for_customer(customer_id: str, artifacts: dict, top_k: int = 12) -> list[dict]`
  - Generate candidates for one customer.
  - Build features.
  - Score candidates.
  - Return ranked recommendation dictionaries.

- `recommend_for_customers(customer_ids: list[str], artifacts: dict, top_k: int = 12) -> pd.DataFrame`
  - Batch version of single-customer inference.

- `fallback_recommendations(artifacts: dict, top_k: int = 12) -> list[dict]`
  - Return popularity-based recommendations for unknown or cold-start users.

Design reasoning:

Inference should not be tied directly to FastAPI. Keeping it separate makes it usable in notebooks, batch jobs, tests, and the API.

### 4.14 `src/inference/postprocess.py`

Purpose:

Format raw model outputs into user-facing recommendation results.

Planned functions:

- `attach_article_metadata(recommendations: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame`
  - Join article names, product type, color, department, and description.

- `format_recommendation_response(recommendations: pd.DataFrame) -> list[dict]`
  - Convert ranked DataFrame rows into API response dictionaries.

### 4.15 `src/api/schemas.py`

Purpose:

Define API request and response contracts.

Planned Pydantic models:

- `RecommendationRequest`
  - Fields:
    - `customer_id: str`
    - `top_k: int = 12`

- `BatchRecommendationRequest`
  - Fields:
    - `customer_ids: list[str]`
    - `top_k: int = 12`

- `RecommendationItem`
  - Fields:
    - `article_id: int | str`
    - `score: float`
    - `rank: int`
    - optional article metadata fields

- `RecommendationResponse`
  - Fields:
    - `customer_id: str`
    - `recommendations: list[RecommendationItem]`

Design reasoning:

Pydantic schemas make the serving interface explicit and testable.

### 4.16 `src/api/dependencies.py`

Purpose:

Provide shared API dependencies.

Planned functions:

- `get_artifacts() -> dict`
  - Return loaded model artifacts.
  - Used by API endpoints.

Design reasoning:

The API should load heavy artifacts once and reuse them across requests.

### 4.17 `src/api/main.py`

Purpose:

Expose the recommender through FastAPI.

Planned endpoints:

- `GET /health`
  - Return API health and artifact load status.

- `POST /recommendations`
  - Accept one `customer_id`.
  - Return top-k recommendations.

- `POST /recommendations/batch`
  - Accept multiple customer IDs.
  - Return recommendations for each.

Planned functions:

- `startup_event() -> None`
  - Load recommender artifacts at API startup.

- `health() -> dict`
  - Return service status.

- `recommend(request: RecommendationRequest) -> RecommendationResponse`
  - Serve one customer.

- `recommend_batch(request: BatchRecommendationRequest) -> list[RecommendationResponse]`
  - Serve multiple customers.

### 4.18 `src/utils/config.py`

Purpose:

Centralize configuration loading and path resolution.

Planned functions:

- `load_config(config_path: str | Path) -> dict`
  - Load YAML config.

- `get_project_root() -> Path`
  - Return repository root path.

- `resolve_path(path: str | Path) -> Path`
  - Resolve relative project paths consistently.

### 4.19 `src/utils/logging.py`

Purpose:

Create consistent logging across scripts and services.

Planned functions:

- `get_logger(name: str) -> logging.Logger`
  - Configure and return a logger.

Design reasoning:

Logging is necessary for debugging long-running data and training jobs.

## 5. Configuration And Artifacts

### 5.1 `configs/config.yaml`

Purpose:

Store paths and project settings.

Planned fields:

```yaml
data:
  raw_dir: data/raw
  processed_dir: data/processed
  transactions_file: transactions_train.parquet
  articles_file: articles.parquet
  customers_file: customers.parquet

splits:
  validation_days: 7
  test_days: 7

recommendation:
  top_k: 12
  candidate_top_k: 100

artifacts:
  model_dir: artifacts/models
  feature_dir: artifacts/features
  report_dir: artifacts/reports

random_seed: 42
```

Design reasoning:

Configuration makes the pipeline easier to modify without editing code.

### 5.2 Artifact directories

Planned outputs:

```text
artifacts/
  features/
    feature_columns.json
  models/
    popularity.parquet
    ranker.joblib
  reports/
    training_summary.json
    evaluation_metrics.csv
    evaluation_metrics.json
```

Design reasoning:

Artifacts should be separated from source code and reproducible through training commands.

## 6. Testing Plan

Implemented tests:

- `tests/conftest.py`
  - Adds the project root to `sys.path` for local test imports.

- `tests/test_candidates.py`
  - Candidate generation returns expected columns and customer-item pairs.
  - Recent-popular candidates come from the latest transaction window.
  - User-history candidates keep each customer's most recent unique items.
  - Merged candidates contain no duplicate `(customer_id, article_id)` pairs.
  - Empty candidate source lists raise `ValueError`.
  - Labeled candidates correctly mark known validation purchases.

- `tests/test_metrics.py`
  - Precision, recall, average precision, and NDCG match hand-computed examples.
  - Empty relevant sets return `0.0`.
  - Non-positive `k` raises `ValueError`.

- `tests/test_popularity.py`
  - Popularity model output is sorted by purchase count.
  - Popular recommendations repeat top items for each unique customer.
  - Non-positive `top_k` raises `ValueError`.

- `tests/test_ranker.py`
  - Numeric feature selection excludes IDs, labels, sources, and dates.
  - Ranker training rejects single-class labels.
  - Recommendations are sorted per customer and limited to top `k`.
  - Non-positive `top_k` raises `ValueError`.

Planned tests:

### `tests/test_validate_data.py`

Tests:

- validators pass on valid mini DataFrames.
- validators raise `ValueError` when required columns are missing.

### `tests/test_preprocess_data.py`

Tests:

- transaction dates are converted to datetime.
- duplicates are removed.
- transactions are sorted chronologically.
- article/customer categorical nulls are filled.
- customer age nulls are filled with a numeric value.

### `tests/test_split_data.py`

Tests:

- validation rows are later than train rows.
- empty input raises `ValueError`.
- non-positive `validation_days` raises `ValueError`.

### `tests/test_build_features.py`

Tests:

- user feature output contains expected columns.
- item feature output contains expected columns.
- popularity features are sorted descending.
- popularity rank starts at 1.

### `tests/test_api.py`

Tests:

- `/health` returns success.
- `/recommendations` returns the expected response shape with mocked artifacts.

Design reasoning:

Tests should use small synthetic DataFrames rather than the full H&M dataset. This keeps tests fast and deterministic.

## 7. Production Packaging

### `Dockerfile`

Purpose:

Package the API service.

Responsibilities:

- Install Python dependencies.
- Copy source code.
- Expose the FastAPI app.
- Start `uvicorn`.

### `docker-compose.yml`

Purpose:

Run the recommender API locally with one command.

Planned service:

- `api`
  - Builds from `Dockerfile`.
  - Mounts or includes artifacts.
  - Serves FastAPI.

### `.env.example`

Purpose:

Document environment variables.

Planned variables:

```text
CONFIG_PATH=configs/config.yaml
MODEL_PATH=artifacts/models/ranker.joblib
```

### `Makefile`

Purpose:

Provide simple project commands.

Planned commands:

- `make setup`
- `make dataset`
- `make train`
- `make evaluate`
- `make api`
- `make test`

Design reasoning:

Simple commands make the project easier for recruiters, reviewers, and future agents to run.

## 8. Final End-To-End Flow

The final project should run as:

```powershell
pip install -r requirements.txt
python -m src.data.make_dataset
python -m src.training.train_baseline
python -m src.training.train_ranker
python -m src.training.evaluate_model
uvicorn src.api.main:app --reload
```

Optional Docker flow:

```powershell
docker compose up --build
```

## 9. Learning Path

Recommended implementation order:

1. Use multiple candidate sources in ranker training and evaluation.
2. Add article/customer metadata features to candidate feature tables.
3. Save baseline and ranker artifacts from training scripts.
4. Add configuration loading and project path resolution.
5. Build inference utilities with popularity fallback.
6. Build the FastAPI service.
7. Add persisted evaluation reports.
8. Consider upgrading the logistic ranker to LightGBM.
9. Add Docker and command shortcuts.
10. Keep tests and documentation aligned as each component changes.

This order starts from the current repo state: data loading, preprocessing, splitting, multiple candidate generators, candidate labeling, basic features, metrics, a popularity baseline, and a logistic ranker experiment already exist. The next priority is using the richer candidates in training, then turning those experiments into reusable artifacts and inference paths.

## 10. Acceptance Criteria

The project is complete when:

- raw parquet files load successfully.
- schema validation catches missing required columns.
- preprocessing creates clean transaction, article, and customer tables.
- temporal train, validation, and test splits are saved.
- candidate generation creates labeled candidate rows.
- feature engineering creates model-ready tables.
- popularity baseline produces recommendations.
- ranking model trains and saves artifacts.
- evaluation reports include `Precision@K`, `Recall@K`, `MAP@K`, and `NDCG@K`.
- FastAPI serves recommendations for known and unknown users.
- unknown users receive popularity fallback recommendations.
- tests pass with `pytest`.
- Docker starts the API successfully.
- README and docs explain the project clearly.
