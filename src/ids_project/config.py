from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field
from pathlib import Path

PRODUCTION_PROFILE_NAME = "default-prod"
U2R_SPECIALIST_PROFILE_NAME = "u2r-specialist"

DEFAULT_CLASS_WEIGHTS = {
    "normal": 1.0,
    "dos": 1.0,
    "probe": 1.0,
    "r2l": 1.5,
    "u2r": 2.5,
}

TRAINING_PROFILES: dict[str, dict[str, Any]] = {
    PRODUCTION_PROFILE_NAME: {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "n_estimators": 200,
        "min_child_samples": 20,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "use_smote": True,
        "custom_class_weights": dict(DEFAULT_CLASS_WEIGHTS),
    },
    U2R_SPECIALIST_PROFILE_NAME: {
        "learning_rate": 0.06,
        "num_leaves": 15,
        "max_depth": 8,
        "n_estimators": 260,
        "min_child_samples": 40,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 1.0,
        "use_smote": True,
        "custom_class_weights": {
            "normal": 1.0,
            "dos": 1.0,
            "probe": 1.0,
            "r2l": 1.5,
            "u2r": 2.5,
        },
    },
}


def build_profile_config(profile_name: str) -> dict[str, Any]:
    profile = TRAINING_PROFILES.get(profile_name)
    if profile is None:
        available_profiles = ", ".join(sorted(TRAINING_PROFILES))
        raise ValueError(f"Unknown training profile {profile_name!r}. Available profiles: {available_profiles}.")
    return {key: dict(value) if isinstance(value, dict) else value for key, value in profile.items()}


@dataclass(slots=True)
class PathsConfig:
    dataset_path: Path
    artifact_dir: Path = Path("artifacts/latest")
    report_dir: Path = Path("reports/latest")

    def resolve(self) -> "PathsConfig":
        return PathsConfig(
            dataset_path=self.dataset_path.expanduser().resolve(),
            artifact_dir=self.artifact_dir.expanduser().resolve(),
            report_dir=self.report_dir.expanduser().resolve(),
        )


@dataclass(slots=True)
class TrainingConfig:
    dataset_path: Path
    test_size: float = 0.2
    validation_size: float = 0.25
    random_state: int = 42
    positive_label: str = "normal"
    profile_name: str = PRODUCTION_PROFILE_NAME
    artifact_dir: Path = Path("artifacts/latest")
    report_dir: Path = Path("reports/latest")
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    n_estimators: int = 200
    min_child_samples: int = 20
    feature_fraction: float = 1.0
    bagging_fraction: float = 1.0
    bagging_freq: int = 0
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    threshold: float = 0.5
    early_stopping_rounds: int = 30
    categorical_min_frequency: int = 2
    numeric_clip_quantile: float = 0.995
    target_column: str = "label"
    difficulty_column: str = "difficulty"
    report_top_features: int = 15
    report_precision_digits: int = 4
    use_gpu: bool = False
    require_gpu: bool = False
    allow_gpu_fallback: bool = True
    gpu_platform_id: int = 0
    gpu_device_id: int = 0
    progress_bar: bool = True
    use_smote: bool = True
    custom_class_weights: dict[str, float] | None = field(default_factory=lambda: dict(DEFAULT_CLASS_WEIGHTS))
    notes: str = ""
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def paths(self) -> PathsConfig:
        return PathsConfig(
            dataset_path=self.dataset_path,
            artifact_dir=self.artifact_dir,
            report_dir=self.report_dir,
        ).resolve()
