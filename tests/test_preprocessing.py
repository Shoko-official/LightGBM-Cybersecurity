from __future__ import annotations

import numpy as np
import pytest

from ids_project.config import TrainingConfig
from ids_project.data.dataset import load_dataset
from ids_project.preprocessing import fit_preprocessing, transform_features, transform_labels
from ids_project.training import _balance_dataset


def test_preprocessing_is_deterministic(sample_dataset_path):
    dataset = load_dataset(TrainingConfig(dataset_path=sample_dataset_path))
    features = dataset.drop(columns=["label"])
    labels = dataset["label"]

    train_matrix, encoded_labels, artifacts = fit_preprocessing(features, labels, TrainingConfig(dataset_path=sample_dataset_path))
    second_matrix = transform_features(features, artifacts)
    second_labels = transform_labels(labels, artifacts)

    np.testing.assert_allclose(train_matrix, second_matrix)
    np.testing.assert_array_equal(encoded_labels.to_numpy(), second_labels.to_numpy())


def test_preprocessing_rejects_missing_columns(sample_dataset_path):
    dataset = load_dataset(TrainingConfig(dataset_path=sample_dataset_path))
    features = dataset.drop(columns=["label", "service"])
    labels = dataset["label"]

    with pytest.raises(ValueError, match="missing columns"):
        fit_preprocessing(features, labels, TrainingConfig(dataset_path=sample_dataset_path))


def test_preprocessing_can_disable_anomaly_feature(sample_dataset_path):
    dataset = load_dataset(TrainingConfig(dataset_path=sample_dataset_path))
    features = dataset.drop(columns=["label"])
    labels = dataset["label"]

    _, _, artifacts = fit_preprocessing(
        features,
        labels,
        TrainingConfig(dataset_path=sample_dataset_path, use_anomaly_feature=False),
    )

    assert "anomaly_extractor" not in artifacts.pipeline.named_steps
    assert "unsupervised_anomaly_score" not in artifacts.feature_names


def test_training_config_rejects_invalid_categorical_frequency(sample_dataset_path):
    with pytest.raises(ValueError, match="categorical_min_frequency"):
        TrainingConfig(dataset_path=sample_dataset_path, categorical_min_frequency=0)


def test_balancing_duplicates_existing_rows_without_synthetic_values(sample_dataset_path):
    features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    labels = np.array([0, 1, 1])

    balanced_features, balanced_labels = _balance_dataset(
        features,
        labels,
        TrainingConfig(dataset_path=sample_dataset_path),
    )

    original_rows = {tuple(row) for row in features.tolist()}
    assert len(balanced_features) == 4
    assert set(balanced_labels.tolist()) == {0, 1}
    assert all(tuple(row) in original_rows for row in balanced_features.tolist())
