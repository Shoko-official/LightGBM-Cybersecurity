from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ids_project.config import TrainingConfig
from ids_project.data.dataset import build_split, load_dataset


def test_load_dataset_normalizes_schema(sample_dataset_path):
    config = TrainingConfig(dataset_path=sample_dataset_path)
    dataset = load_dataset(config)
    assert "difficulty" not in dataset.columns
    assert "label" in dataset.columns
    assert len(dataset) == 60


def test_load_dataset_supports_headerless_nsl_kdd(sample_dataset_frame, tmp_path):
    path = tmp_path / "nsl_kdd_raw.txt"
    sample_dataset_frame.to_csv(path, index=False, header=False)

    dataset = load_dataset(TrainingConfig(dataset_path=path))

    assert len(dataset) == len(sample_dataset_frame)
    assert "difficulty" not in dataset.columns
    assert set(dataset["label"].unique()) == {
        "normal",
        "neptune",
        "satan",
        "guess_passwd",
        "buffer_overflow",
    }


def test_load_dataset_rejects_unknown_columns(sample_dataset_frame, tmp_path):
    invalid = sample_dataset_frame.copy()
    invalid["unexpected"] = 1
    path = tmp_path / "invalid.csv"
    invalid.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Unexpected columns"):
        load_dataset(TrainingConfig(dataset_path=path))


def test_build_split_is_stratified(sample_dataset_path):
    config = TrainingConfig(dataset_path=sample_dataset_path)
    dataset = load_dataset(config)
    split = build_split(dataset, config)
    expected_labels = {"normal", "neptune", "satan", "guess_passwd", "buffer_overflow"}
    assert set(split.labels_train.unique()) == expected_labels
    assert set(split.labels_validation.unique()) == expected_labels
    assert set(split.labels_test.unique()) == expected_labels


def test_build_split_falls_back_for_singleton_classes(sample_dataset_frame):
    frame = sample_dataset_frame.copy()
    singleton_label = "buffer_overflow"
    singleton_rows = frame[frame["label"] == singleton_label].head(1)
    other_rows = frame[frame["label"] != singleton_label]
    frame = pd.concat([other_rows, singleton_rows], ignore_index=True)

    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        split = build_split(frame.drop(columns=["difficulty"]), TrainingConfig(dataset_path=Path("unused.csv")))

    warning_messages = [str(w.message) for w in caught]
    assert any("non\u2011stratified" in m or "non-stratified" in m or "<2 rows" in m for m in warning_messages)
    # Split should still succeed
    assert len(split.labels_train) > 0
