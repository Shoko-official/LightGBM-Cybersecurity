from __future__ import annotations

from dataclasses import dataclass

from ids_project.config import TrainingConfig


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    estimator: object
    supports_callbacks: bool


def build_lightgbm(
    config: TrainingConfig,
    classes: list[str],
    *,
    use_gpu: bool | None = None,
) -> ModelSpec:
    use_gpu = config.use_gpu if use_gpu is None else use_gpu

    try:
        from lightgbm import LGBMClassifier
        from sklearn.linear_model import LogisticRegression
    except ModuleNotFoundError:
        return ModelSpec(
            name="lightgbm_fallback_logistic_regression",
            estimator=LogisticRegression(
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
            ),
            supports_callbacks=False,
        )

    class_weight: str | dict[int, float] = "balanced"
    if config.custom_class_weights and classes:
        class_weight = {
            index: config.custom_class_weights.get(label, 1.0)
            for index, label in enumerate(classes)
        }

    estimator_kwargs = {
        "objective": "multiclass",
        "num_class": max(2, len(classes)),
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "n_estimators": config.n_estimators,
        "min_child_samples": config.min_child_samples,
        "random_state": config.random_state,
        "class_weight": class_weight,
        "verbosity": -1,
    }
    if use_gpu:
        estimator_kwargs["device"] = "gpu"
        estimator_kwargs["gpu_platform_id"] = config.gpu_platform_id
        estimator_kwargs["gpu_device_id"] = config.gpu_device_id

    return ModelSpec(
        name="lightgbm",
        estimator=LGBMClassifier(**estimator_kwargs),
        supports_callbacks=True,
    )
