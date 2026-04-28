from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CLASS_WEIGHTS = {
    "normal": 1.0,
    "dos": 1.0,
    "probe": 1.0,
    "r2l": 1.5,
    "u2r": 2.5,
}


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
