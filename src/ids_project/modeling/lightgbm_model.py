from __future__ import annotations

from dataclasses import dataclass

from ids_project.config import TrainingConfig


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    estimator: object
    supports_callbacks: bool


def build_lightgbm(config: TrainingConfig) -> ModelSpec:
    try:
        from lightgbm import LGBMClassifier
        from sklearn.linear_model import LogisticRegression
    except ModuleNotFoundError:
        return ModelSpec(
            name="logistic_regression",
            estimator=LogisticRegression(
                class_weight="balanced",
                solver="lbfgs",
                max_iter=300,
            ),
            supports_callbacks=False,
        )

    label_classes = config.extra_metadata.get("classes", [])
    estimator_kwargs = {
        "objective": "multiclass",
        "num_class": max(2, len(label_classes)),
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "n_estimators": config.n_estimators,
        "min_child_samples": config.min_child_samples,
        "random_state": config.random_state,
        "class_weight": "balanced",
        "verbosity": -1,
    }
    if config.use_gpu:
        estimator_kwargs["device"] = "gpu"
        estimator_kwargs["gpu_platform_id"] = config.gpu_platform_id
        estimator_kwargs["gpu_device_id"] = config.gpu_device_id

    return ModelSpec(
        name="lightgbm",
        estimator=LGBMClassifier(**estimator_kwargs),
        supports_callbacks=True,
    )
