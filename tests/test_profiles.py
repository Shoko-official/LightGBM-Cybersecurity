from __future__ import annotations

import pytest

from ids_project.config import (
    PRODUCTION_PROFILE_NAME,
    TrainingConfig,
    U2R_SPECIALIST_PROFILE_NAME,
    build_profile_config,
)


def test_build_profile_config_returns_production_defaults():
    profile = build_profile_config(PRODUCTION_PROFILE_NAME)

    assert profile["n_estimators"] == 320
    assert profile["num_leaves"] == 7
    assert profile["max_depth"] == 5
    assert profile["threshold"] == 0.30
    assert profile["use_smote"] is False
    assert profile["custom_class_weights"] is None


def test_build_profile_config_returns_specialist_defaults():
    profile = build_profile_config(U2R_SPECIALIST_PROFILE_NAME)

    assert profile["n_estimators"] == 260
    assert profile["num_leaves"] == 15
    assert profile["max_depth"] == 8


def test_build_profile_config_rejects_unknown_profile():
    with pytest.raises(ValueError, match="Unknown training profile"):
        build_profile_config("unknown-profile")


def test_training_config_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="threshold"):
        TrainingConfig(dataset_path="data/raw/nsl_kdd.csv", threshold=1.5)


def test_training_config_rejects_require_gpu_with_fallback():
    with pytest.raises(ValueError, match="CPU fallback"):
        TrainingConfig(
            dataset_path="data/raw/nsl_kdd.csv",
            use_gpu=True,
            require_gpu=True,
            allow_gpu_fallback=True,
        )
