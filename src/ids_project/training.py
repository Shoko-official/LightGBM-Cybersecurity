from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from ids_project.artifacts import save_runtime_bundle
from ids_project.config import TrainingConfig
from ids_project.contracts import ArtifactManifest, ModelMetrics, TrainingResult
from ids_project.data.dataset import build_split, load_dataset
from ids_project.evaluation import build_evaluation_report, save_report
from ids_project.modeling.baselines import build_baseline_classifier
from ids_project.modeling.lightgbm_model import build_lightgbm
from ids_project.preprocessing import fit_preprocessing, transform_features, transform_labels


def train(config: TrainingConfig) -> TrainingResult:
    print("Training process started")
    progress = tqdm(
        total=100,
        desc="Training",
        disable=not config.progress_bar,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )

    progress.set_description("Loading dataset")
    dataset = load_dataset(config)
    split = build_split(dataset, config)
    progress.update(10)

    progress.set_description("Fitting preprocessing")
    train_matrix, train_labels, preprocessing = fit_preprocessing(
        split.features_train,
        split.labels_train,
        config,
    )
    validation_matrix = transform_features(split.features_validation, preprocessing)
    validation_labels = transform_labels(split.labels_validation, preprocessing)
    progress.update(40)

    progress.set_description("Evaluating baseline")
    baseline_model = build_baseline_classifier()
    train_frame = pd.DataFrame(train_matrix, columns=preprocessing.feature_names)
    validation_frame = pd.DataFrame(validation_matrix, columns=preprocessing.feature_names)
    baseline_model.fit(train_frame, train_labels)
    baseline_report = build_evaluation_report(
        model=baseline_model,
        features=validation_frame,
        labels=validation_labels,
        feature_names=preprocessing.feature_names,
        model_name="logistic_regression",
        threshold=config.threshold,
        split_name="validation",
        top_k_features=config.report_top_features,
        precision_digits=config.report_precision_digits,
        original_labels=split.labels_validation,
    )
    progress.update(10)

    progress.set_description("Training main model")
    classes = list(preprocessing.label_encoder.classes_)
    metadata = dict(config.extra_metadata)
    metadata["classes"] = classes
    balanced_matrix, balanced_labels = _balance_dataset(train_matrix, train_labels, config)
    balanced_frame = pd.DataFrame(balanced_matrix, columns=preprocessing.feature_names)
    model_spec = build_lightgbm(config, classes)
    model = model_spec.estimator

    try:
        model.fit(balanced_frame, balanced_labels)
    except Exception as exc:
        if config.use_gpu and model_spec.name == "lightgbm":
            print(f"GPU training failed, retrying on CPU: {exc}")
            model_spec = build_lightgbm(config, classes, use_gpu=False)
            model = model_spec.estimator
            model.fit(balanced_frame, balanced_labels)
            metadata["gpu_fallback_reason"] = str(exc)
        else:
            raise
    progress.update(30)

    progress.set_description("Finalizing report")
    validation_report = build_evaluation_report(
        model=model,
        features=validation_frame,
        labels=validation_labels,
        feature_names=preprocessing.feature_names,
        model_name=model_spec.name,
        threshold=config.threshold,
        split_name="validation",
        top_k_features=config.report_top_features,
        precision_digits=config.report_precision_digits,
        original_labels=split.labels_validation,
    )
    report_path = save_report(validation_report, config.paths.report_dir, "validation_report.json")

    union = preprocessing.pipeline.named_steps["union"]
    manifest = ArtifactManifest(
        model_name=model_spec.name,
        dataset_path=str(config.paths.dataset_path),
        target_column=config.target_column,
        threshold=config.threshold,
        random_state=config.random_state,
        feature_columns=preprocessing.feature_names,
        categorical_columns=list(union.transformers_[2][2]),
        numeric_columns=list(union.transformers_[0][2]) + list(union.transformers_[1][2]),
        label_mapping=preprocessing.label_encoder.mapping,
        baseline_metrics={"logistic_regression": _metrics_to_dict(baseline_report.metrics)},
        validation_metrics=_metrics_to_dict(validation_report.metrics),
        files={
            "model": "model.joblib",
            "preprocessor": "preprocessor.joblib",
            "report": str(report_path),
            "manifest": "manifest.json",
        },
        metadata={"notes": config.notes, **metadata},
    )
    _, _, manifest_path = save_runtime_bundle(config.paths.artifact_dir, preprocessing, model, manifest)
    progress.update(10)
    progress.close()

    return TrainingResult(
        model_name=model_spec.name,
        artifact_dir=config.paths.artifact_dir,
        report_path=report_path,
        manifest_path=manifest_path,
        threshold=config.threshold,
        baseline_metrics={"logistic_regression": baseline_report.metrics},
        validation_report=validation_report,
    )


def _metrics_to_dict(metrics: ModelMetrics) -> dict[str, float]:
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
        "roc_auc": metrics.roc_auc,
        "average_precision": metrics.average_precision,
    }


def _balance_dataset(
    features: np.ndarray,
    labels: pd.Series,
    config: TrainingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels)

    if config.use_smote:
        try:
            from imblearn.over_sampling import SMOTE

            sampler = SMOTE(random_state=config.random_state)
            return sampler.fit_resample(features, labels_array)
        except ImportError:
            print("imbalanced-learn is not installed. Falling back to random oversampling.")

    unique_labels, counts = np.unique(labels_array, return_counts=True)
    max_count = int(np.max(counts))
    expanded_features = [features]
    expanded_labels = [labels_array]

    for label, count in zip(unique_labels, counts):
        if count >= max_count:
            continue
        target_count = int(count + (max_count - count) * 0.5)
        extra_needed = target_count - int(count)
        if extra_needed <= 0:
            continue
        label_indices = np.where(labels_array == label)[0]
        sampled_indices = np.random.choice(label_indices, size=extra_needed, replace=True)
        expanded_features.append(features[sampled_indices])
        expanded_labels.append(labels_array[sampled_indices])

    return np.vstack(expanded_features), np.concatenate(expanded_labels)
