from __future__ import annotations

import pandas as pd

from ids_project.artifacts import load_runtime_bundle
from ids_project.contracts import BatchPredictionResult, PredictionResult, RuntimeBundle
from ids_project.decision import apply_attack_threshold, model_class_indices
from ids_project.preprocessing import transform_features


def load_runtime(path: str) -> RuntimeBundle:
    return load_runtime_bundle(path)


def predict_one(bundle: RuntimeBundle, payload: dict[str, object]) -> PredictionResult:
    frame = pd.DataFrame([payload])
    transformed = transform_features(frame, bundle.preprocessor)
    transformed_frame = pd.DataFrame(transformed, columns=bundle.manifest.feature_columns)
    probabilities = bundle.model.predict_proba(transformed_frame)
    decisions = _decide(bundle, probabilities)
    predicted_index = int(decisions.predicted_indices[0])
    category = _resolve_category(bundle, predicted_index)
    return PredictionResult(
        label="normal" if category == "normal" else "attack",
        category=category,
        score=float(decisions.class_confidences[0]),
        threshold=bundle.manifest.threshold,
    )


def predict_batch(bundle: RuntimeBundle, payloads: list[dict[str, object]]) -> BatchPredictionResult:
    if not payloads:
        raise ValueError("Prediction payload list cannot be empty.")
    frame = pd.DataFrame(payloads)
    transformed = transform_features(frame, bundle.preprocessor)
    transformed_frame = pd.DataFrame(transformed, columns=bundle.manifest.feature_columns)
    predictions: list[PredictionResult] = []
    probabilities = bundle.model.predict_proba(transformed_frame)
    decisions = _decide(bundle, probabilities)
    for predicted_index, confidence in zip(decisions.predicted_indices, decisions.class_confidences):
        category = _resolve_category(bundle, predicted_index)
        predictions.append(
            PredictionResult(
                label="normal" if category == "normal" else "attack",
                category=category,
                score=float(confidence),
                threshold=bundle.manifest.threshold,
            )
        )
    return BatchPredictionResult(predictions=predictions)


def describe_runtime(bundle: RuntimeBundle) -> dict[str, object]:
    profile_name = bundle.manifest.metadata.get("profile_name", "unknown")
    gpu_backend = bundle.manifest.metadata.get("gpu_backend", "cpu")
    return {
        "model_name": bundle.manifest.model_name,
        "profile_name": profile_name,
        "gpu_backend": gpu_backend,
        "threshold": bundle.manifest.threshold,
        "dataset_path": bundle.manifest.dataset_path,
        "feature_count": len(bundle.manifest.feature_columns),
        "feature_columns": bundle.manifest.feature_columns,
        "label_mapping": bundle.manifest.label_mapping,
    }


def _resolve_category(bundle: RuntimeBundle, predicted_index: int) -> str:
    reverse_mapping = {value: key for key, value in bundle.manifest.label_mapping.items()}
    return reverse_mapping.get(predicted_index, "unknown")


def _decide(bundle: RuntimeBundle, probabilities):
    class_indices = model_class_indices(bundle.model, probabilities.shape[1])
    normal_index = int(bundle.manifest.label_mapping.get("normal", 0))
    return apply_attack_threshold(
        probabilities,
        class_indices,
        normal_index=normal_index,
        threshold=bundle.manifest.threshold,
    )
