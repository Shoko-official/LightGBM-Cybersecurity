from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import platform
from typing import Any

PRODUCTION_PROFILE_NAME = "default-prod"
U2R_SPECIALIST_PROFILE_NAME = "u2r-specialist"
VALID_GPU_BACKENDS = {"auto", "gpu", "cuda"}

DEFAULT_CLASS_WEIGHTS = {
    "normal": 1.0,
    "dos": 1.0,
    "probe": 1.0,
    "r2l": 1.5,
    "u2r": 2.5,
}

TRAINING_PROFILES: dict[str, dict[str, Any]] = {
    PRODUCTION_PROFILE_NAME: {
        "learning_rate": 0.04,
        "num_leaves": 7,
        "max_depth": 5,
        "n_estimators": 320,
        "min_child_samples": 80,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "lambda_l1": 0.8,
        "lambda_l2": 3.0,
        "threshold": 0.30,
        "use_smote": False,
        "custom_class_weights": None,
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


def resolve_gpu_backend(gpu_backend: str) -> str:
    if gpu_backend not in VALID_GPU_BACKENDS:
        available_backends = ", ".join(sorted(VALID_GPU_BACKENDS))
        raise ValueError(f"Unknown GPU backend {gpu_backend!r}. Available backends: {available_backends}.")
    if gpu_backend != "auto":
        return gpu_backend
    return "cuda" if platform.system().lower() == "linux" else "gpu"


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
    gpu_backend: str = "auto"
    require_gpu: bool = False
    allow_gpu_fallback: bool = True
    gpu_platform_id: int = 0
    gpu_device_id: int = 0
    progress_bar: bool = True
    use_smote: bool = True
    custom_class_weights: dict[str, float] | None = field(default_factory=lambda: dict(DEFAULT_CLASS_WEIGHTS))
    notes: str = ""
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def paths(self) -> PathsConfig:
        return PathsConfig(
            dataset_path=self.dataset_path,
            artifact_dir=self.artifact_dir,
            report_dir=self.report_dir,
        ).resolve()

    def validate(self) -> None:
        _validate_ratio("test_size", self.test_size, lower=0.0, upper=1.0, include_upper=False)
        _validate_ratio("validation_size", self.validation_size, lower=0.0, upper=1.0, include_upper=False)
        _validate_ratio("threshold", self.threshold, lower=0.0, upper=1.0, include_lower=True, include_upper=True)
        _validate_ratio(
            "numeric_clip_quantile",
            self.numeric_clip_quantile,
            lower=0.5,
            upper=1.0,
            include_upper=False,
        )
        _validate_ratio("feature_fraction", self.feature_fraction, lower=0.0, upper=1.0, include_upper=True)
        _validate_ratio("bagging_fraction", self.bagging_fraction, lower=0.0, upper=1.0, include_upper=True)
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0.")
        if self.num_leaves <= 1:
            raise ValueError("num_leaves must be greater than 1.")
        if self.min_child_samples <= 0:
            raise ValueError("min_child_samples must be greater than 0.")
        if self.early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be greater than 0.")
        if self.report_top_features <= 0:
            raise ValueError("report_top_features must be greater than 0.")
        if self.report_precision_digits < 0:
            raise ValueError("report_precision_digits must be non-negative.")
        if self.gpu_platform_id < 0 or self.gpu_device_id < 0:
            raise ValueError("GPU platform and device ids must be non-negative.")
        if self.require_gpu and not self.use_gpu:
            raise ValueError("require_gpu=True requires use_gpu=True.")
        if self.require_gpu and self.allow_gpu_fallback:
            raise ValueError("require_gpu=True cannot allow CPU fallback.")
        if self.custom_class_weights:
            invalid_weights = {
                label: weight
                for label, weight in self.custom_class_weights.items()
                if weight <= 0
            }
            if invalid_weights:
                labels = ", ".join(sorted(invalid_weights))
                raise ValueError(f"Class weights must be greater than 0 for: {labels}.")


def _validate_ratio(
    name: str,
    value: float,
    *,
    lower: float,
    upper: float,
    include_lower: bool = False,
    include_upper: bool = False,
) -> None:
    lower_ok = value >= lower if include_lower else value > lower
    upper_ok = value <= upper if include_upper else value < upper
    if not lower_ok or not upper_ok:
        left = "[" if include_lower else "("
        right = "]" if include_upper else ")"
        raise ValueError(f"{name} must be in range {left}{lower}, {upper}{right}.")
