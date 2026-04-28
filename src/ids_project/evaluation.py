from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ids_project.artifacts import ensure_directory
from ids_project.contracts import EvaluationReport, ModelMetrics, RuntimeBundle


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
) -> EvaluationReport:
    if isinstance(features, np.ndarray) and len(feature_names) == features.shape[1]:
        features_df = pd.DataFrame(features, columns=feature_names)
        probabilities = model.predict_proba(features_df)
    else:
        probabilities = model.predict_proba(features)

    label_values = np.asarray(labels)
    predictions = np.argmax(probabilities, axis=1)
    is_binary = probabilities.shape[1] == 2 and len(np.unique(label_values)) == 2
    if is_binary:
        positive_scores = probabilities[:, 1]
        roc_auc = float(roc_auc_score(label_values, positive_scores))
        average_precision = float(average_precision_score(label_values, positive_scores))
    else:
        roc_auc = float(
            roc_auc_score(
                label_values,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=list(range(probabilities.shape[1])),
            )
        )
        average_precision = 0.0

    metrics = ModelMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, average="macro", zero_division=0)),
        recall=float(recall_score(labels, predictions, average="macro", zero_division=0)),
        f1_score=float(f1_score(labels, predictions, average="macro", zero_division=0)),
        roc_auc=roc_auc,
        average_precision=average_precision,
    )
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, predictions).tolist()
    return EvaluationReport(
        model_name=model_name,
        threshold=threshold,
        metrics=metrics,
        confusion_matrix=matrix,
        classification_report=_round_report(report, precision_digits),
        top_features=_extract_top_features(model, feature_names, top_k_features),
        split_name=split_name,
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
    )


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
