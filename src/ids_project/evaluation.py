from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from ids_project.artifacts import ensure_directory
from ids_project.contracts import EvaluationReport, ModelMetrics, RuntimeBundle
from ids_project.decision import apply_attack_threshold, model_class_indices


def build_evaluation_report(
    model: Any,
    features: Any,
    labels: Any,
    feature_names: list[str],
    model_name: str,
    threshold: float,
    split_name: str,
    top_k_features: int,
    precision_digits: int,
    original_labels: pd.Series | None = None,
    label_mapping: dict[str, int] | None = None,
    normal_label: str = "normal",
) -> EvaluationReport:
    if isinstance(features, np.ndarray) and len(feature_names) == features.shape[1]:
        features_df = pd.DataFrame(features, columns=feature_names)
        probabilities = model.predict_proba(features_df)
    else:
        probabilities = model.predict_proba(features)

    label_values = np.asarray(labels)
    class_indices = model_class_indices(model, probabilities.shape[1])
    normal_index = _normal_index(label_mapping, normal_label)
    decisions = apply_attack_threshold(
        probabilities,
        class_indices,
        normal_index=normal_index,
        threshold=threshold,
    )
    predictions = decisions.predicted_indices
    report_labels = _report_labels(label_values, predictions, class_indices)
    class_labels = _label_names(report_labels, label_mapping)
    true_attack = label_values != normal_index
    predicted_attack = predictions != normal_index

    metrics = ModelMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(label_values, predictions)),
        precision=float(
            precision_score(
                label_values,
                predictions,
                labels=report_labels,
                average="macro",
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                label_values,
                predictions,
                labels=report_labels,
                average="macro",
                zero_division=0,
            )
        ),
        f1_score=float(
            f1_score(
                label_values,
                predictions,
                labels=report_labels,
                average="macro",
                zero_division=0,
            )
        ),
        roc_auc=_macro_roc_auc(label_values, probabilities, class_indices),
        average_precision=_macro_average_precision(label_values, probabilities, class_indices),
        attack_precision=float(precision_score(true_attack, predicted_attack, zero_division=0)),
        attack_recall=float(recall_score(true_attack, predicted_attack, zero_division=0)),
        attack_f1_score=float(f1_score(true_attack, predicted_attack, zero_division=0)),
        attack_roc_auc=_safe_binary_roc_auc(true_attack, decisions.attack_scores),
        attack_average_precision=_safe_average_precision(true_attack, decisions.attack_scores),
    )
    report = classification_report(
        label_values,
        predictions,
        labels=report_labels,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(label_values, predictions, labels=report_labels).tolist()
    return EvaluationReport(
        model_name=model_name,
        threshold=threshold,
        metrics=metrics,
        confusion_matrix=matrix,
        classification_report=_round_report(report, precision_digits),
        top_features=_extract_top_features(model, feature_names, top_k_features),
        split_name=split_name,
        class_labels=class_labels,
        attack_category_recall=_calculate_category_recall(labels, predictions, original_labels),
    )


def save_report(report: EvaluationReport, report_dir: Path, filename: str) -> Path:
    target_dir = ensure_directory(report_dir)
    path = target_dir / filename
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path


def evaluate(bundle: RuntimeBundle, split: tuple[Any, Any], split_name: str = "evaluation") -> EvaluationReport:
    features, labels = split
    transformed = bundle.preprocessor.pipeline.transform(features)
    encoded_labels = bundle.preprocessor.label_encoder.transform(labels)
    return build_evaluation_report(
        model=bundle.model,
        features=transformed,
        labels=encoded_labels,
        feature_names=bundle.manifest.feature_columns,
        model_name=bundle.manifest.model_name,
        threshold=bundle.manifest.threshold,
        split_name=split_name,
        top_k_features=min(15, len(bundle.manifest.feature_columns)),
        precision_digits=4,
        original_labels=labels,
        label_mapping=bundle.manifest.label_mapping,
    )


def _normal_index(label_mapping: dict[str, int] | None, normal_label: str) -> int:
    if label_mapping is None:
        return 0
    return int(label_mapping.get(normal_label, 0))


def _report_labels(
    label_values: np.ndarray,
    predictions: np.ndarray,
    class_indices: np.ndarray,
) -> list[int]:
    values = set(np.asarray(label_values, dtype=int).tolist())
    values.update(np.asarray(predictions, dtype=int).tolist())
    values.update(np.asarray(class_indices, dtype=int).tolist())
    return sorted(values)


def _label_names(labels: list[int], label_mapping: dict[str, int] | None) -> list[str]:
    if label_mapping is None:
        return [str(label) for label in labels]
    reverse_mapping = {value: key for key, value in label_mapping.items()}
    return [
        reverse_mapping.get(label, "unknown" if label == -1 else str(label))
        for label in labels
    ]


def _macro_roc_auc(
    label_values: np.ndarray,
    probabilities: np.ndarray,
    class_indices: np.ndarray,
) -> float:
    known_labels, known_probabilities = _known_class_probabilities(label_values, probabilities, class_indices)
    if len(class_indices) < 2 or len(np.unique(known_labels)) < 2:
        return 0.0
    y_true = _binarized_labels(known_labels, class_indices)
    scores: list[float] = []
    for index in range(y_true.shape[1]):
        class_truth = y_true[:, index]
        if len(np.unique(class_truth)) < 2:
            continue
        try:
            scores.append(float(roc_auc_score(class_truth, known_probabilities[:, index])))
        except ValueError:
            continue
    return float(np.mean(scores)) if scores else 0.0


def _macro_average_precision(
    label_values: np.ndarray,
    probabilities: np.ndarray,
    class_indices: np.ndarray,
) -> float:
    known_labels, known_probabilities = _known_class_probabilities(label_values, probabilities, class_indices)
    if len(class_indices) < 2 or len(np.unique(known_labels)) < 2:
        return 0.0
    y_true = _binarized_labels(known_labels, class_indices)
    scores: list[float] = []
    for index in range(y_true.shape[1]):
        class_truth = y_true[:, index]
        if not np.any(class_truth):
            continue
        try:
            scores.append(float(average_precision_score(class_truth, known_probabilities[:, index])))
        except ValueError:
            continue
    return float(np.mean(scores)) if scores else 0.0


def _known_class_probabilities(
    label_values: np.ndarray,
    probabilities: np.ndarray,
    class_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(label_values, class_indices)
    return label_values[mask], probabilities[mask]


def _binarized_labels(label_values: np.ndarray, class_indices: np.ndarray) -> np.ndarray:
    y_true = label_binarize(label_values, classes=class_indices.tolist())
    if len(class_indices) == 2 and y_true.shape[1] == 1:
        return np.column_stack([1 - y_true[:, 0], y_true[:, 0]])
    return y_true


def _safe_binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.0


def _safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    try:
        return float(average_precision_score(labels, scores))
    except ValueError:
        return 0.0


def _extract_top_features(model: Any, feature_names: list[str], top_k_features: int) -> list[dict[str, float]]:
    if not hasattr(model, "feature_importances_"):
        return []
    importances = model.feature_importances_
    ranked_indices = np.argsort(importances)[::-1][:top_k_features]
    return [
        {"feature": feature_names[index], "importance": float(importances[index])}
        for index in ranked_indices
    ]


def _round_report(report: dict[str, Any], digits: int) -> dict[str, dict[str, float]]:
    rounded: dict[str, dict[str, float]] = {}
    for label, values in report.items():
        if isinstance(values, dict):
            rounded[label] = {key: round(float(value), digits) for key, value in values.items()}
        else:
            rounded[label] = {"value": round(float(values), digits)}
    return rounded


def _calculate_category_recall(
    encoded_labels: Any,
    predictions: Any,
    original_labels: pd.Series | None,
) -> dict[str, float]:
    if original_labels is None:
        return {}
    data = pd.DataFrame(
        {
            "true_encoded": np.asarray(encoded_labels),
            "pred_encoded": np.asarray(predictions),
            "category": original_labels.astype(str).str.strip().str.lower().to_numpy(),
        }
    )
    attacks = data[data["category"] != "normal"]
    if attacks.empty:
        return {}

    recalls: dict[str, float] = {}
    for category in attacks["category"].unique():
        category_rows = attacks[attacks["category"] == category]
        recalls[str(category)] = float(
            (category_rows["pred_encoded"] == category_rows["true_encoded"]).mean()
        )
    return recalls
