from __future__ import annotations

import pytest

from ids_project.config import TrainingConfig
from ids_project.data.dataset import build_split, load_dataset


def test_load_dataset_normalizes_schema(sample_dataset_path):
    config = TrainingConfig(dataset_path=sample_dataset_path)
    dataset = load_dataset(config)
    assert "difficulty" not in dataset.columns
    assert "label" in dataset.columns
    assert len(dataset) == 24


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
    assert set(split.labels_train.unique()) == {"normal", "neptune"}
    assert set(split.labels_validation.unique()) == {"normal", "neptune"}
    assert set(split.labels_test.unique()) == {"normal", "neptune"}
