from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from ids_project.config import TrainingConfig
from ids_project.contracts import CATEGORICAL_COLUMNS, DatasetSummary, NSL_KDD_COLUMNS, SplitData


def load_dataset(config: TrainingConfig) -> pd.DataFrame:
    dataset_path = config.paths.dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    separator = _detect_separator(dataset_path)
    frame = pd.read_csv(dataset_path, sep=separator)
    if _looks_like_headerless_nsl_kdd(frame):
        raw_columns = NSL_KDD_COLUMNS + [config.target_column, config.difficulty_column]
        frame = pd.read_csv(dataset_path, sep=separator, header=None, names=raw_columns)
    return _normalize_dataset(frame, config.target_column, config.difficulty_column, dataset_path.name)


def build_split(frame: pd.DataFrame, config: TrainingConfig) -> SplitData:
    train_idx, test_idx = _stratified_split_indices(frame[config.target_column], config.test_size, config.random_state)
    train_frame = frame.iloc[train_idx].reset_index(drop=True)
    test_frame = frame.iloc[test_idx].reset_index(drop=True)

    inner_train_idx, valid_idx = _stratified_split_indices(
        train_frame[config.target_column],
        config.validation_size,
        config.random_state,
    )
    final_train = train_frame.iloc[inner_train_idx].reset_index(drop=True)
    validation = train_frame.iloc[valid_idx].reset_index(drop=True)

    return SplitData(
        features_train=final_train.drop(columns=[config.target_column]),
        features_validation=validation.drop(columns=[config.target_column]),
        features_test=test_frame.drop(columns=[config.target_column]),
        labels_train=final_train[config.target_column],
        labels_validation=validation[config.target_column],
        labels_test=test_frame[config.target_column],
        summary=DatasetSummary(
            row_count=len(frame),
            feature_count=len(frame.columns) - 1,
            label_distribution={key: int(value) for key, value in frame[config.target_column].value_counts().to_dict().items()},
            source_path=str(config.paths.dataset_path),
        ),
    )


def _stratified_split_indices(labels: pd.Series, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    values = labels.to_numpy()
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in range (0.0, 1.0).")
    counts = labels.value_counts()
    too_small = counts[counts < 2]
    if not too_small.empty:
        classes = ", ".join(str(label) for label in too_small.index)
        raise ValueError(f"Cannot stratify classes with fewer than 2 rows: {classes}.")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(np.zeros(len(values)), values))
    return np.array(sorted(train_idx), dtype=int), np.array(sorted(test_idx), dtype=int)


def _normalize_dataset(frame: pd.DataFrame, target_column: str, difficulty_column: str, dataset_name: str) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [column.strip().lower() for column in normalized.columns]

    expected_columns = NSL_KDD_COLUMNS + [target_column]
    optional_columns = {difficulty_column}
    missing_columns = [column for column in expected_columns if column not in normalized.columns]
    unknown_columns = [
        column for column in normalized.columns if column not in expected_columns and column not in optional_columns
    ]

    if missing_columns:
        raise ValueError(f"Dataset schema mismatch for {dataset_name}. Missing columns: {', '.join(missing_columns)}")
    if unknown_columns:
        raise ValueError(f"Dataset schema mismatch for {dataset_name}. Unexpected columns: {', '.join(unknown_columns)}")

    if difficulty_column in normalized.columns:
        normalized = normalized.drop(columns=[difficulty_column])

    for column in CATEGORICAL_COLUMNS:
        normalized[column] = normalized[column].astype(str).str.strip()

    numeric_columns = [column for column in NSL_KDD_COLUMNS if column not in CATEGORICAL_COLUMNS]
    normalized[numeric_columns] = normalized[numeric_columns].apply(pd.to_numeric, errors="raise")
    normalized[target_column] = normalized[target_column].astype(str).str.strip()

    if normalized[target_column].nunique() < 2:
        raise ValueError("Dataset must contain at least two label classes.")

    return normalized


def _looks_like_headerless_nsl_kdd(frame: pd.DataFrame) -> bool:
    expected_widths = {len(NSL_KDD_COLUMNS) + 1, len(NSL_KDD_COLUMNS) + 2}
    return frame.shape[1] in expected_widths and not set(NSL_KDD_COLUMNS).issubset({str(column).lower() for column in frame.columns})


def _detect_separator(dataset_path) -> str:
    if dataset_path.suffix.lower() == ".tsv":
        return "\t"
    preview = dataset_path.open("r", encoding="utf-8", errors="ignore").readline()
    if "\t" in preview and preview.count("\t") > preview.count(","):
        return "\t"
    return ","
