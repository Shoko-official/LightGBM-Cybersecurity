from __future__ import annotations

import numpy as np
import pytest

from ids_project.config import TrainingConfig
from ids_project.data.dataset import load_dataset
from ids_project.preprocessing import fit_preprocessing, transform_features, transform_labels


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
