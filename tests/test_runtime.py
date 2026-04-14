from __future__ import annotations

import pytest

from ids_project.runtime import load_runtime, predict_batch, predict_one


@pytest.fixture()
def trained_bundle(trained_artifact_dir):
    return load_runtime(trained_artifact_dir)


def test_runtime_predict_one(trained_bundle, sample_dataset_frame):
    payload = sample_dataset_frame.drop(columns=["label", "difficulty"]).iloc[0].to_dict()
    result = predict_one(trained_bundle, payload)
    assert result.label in {"normal", "attack"}
    assert 0.0 <= result.score <= 1.0


def test_runtime_predict_batch_requires_payloads(trained_bundle):
    with pytest.raises(ValueError, match="cannot be empty"):
        predict_batch(trained_bundle, [])


def test_runtime_rejects_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required artifact file not found"):
        load_runtime(tmp_path / "missing")
