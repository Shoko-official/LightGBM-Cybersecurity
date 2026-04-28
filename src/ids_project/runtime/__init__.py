from __future__ import annotations

import pandas as pd

from ids_project.artifacts import load_runtime_bundle
from ids_project.contracts import BatchPredictionResult, PredictionResult, RuntimeBundle
from ids_project.preprocessing import transform_features


def load_runtime(path: str) -> RuntimeBundle:
    return load_runtime_bundle(path)


def predict_one(bundle: RuntimeBundle, payload: dict[str, object]) -> PredictionResult:
    frame = pd.DataFrame([payload])
    transformed = transform_features(frame, bundle.preprocessor)
    transformed_frame = pd.DataFrame(transformed, columns=bundle.manifest.feature_columns)
    probabilities = bundle.model.predict_proba(transformed_frame)[0]
    predicted_index = int(probabilities.argmax())
    category = _resolve_category(bundle, predicted_index)
    return PredictionResult(
        label="normal" if category == "normal" else "attack",
        category=category,
        score=float(probabilities[predicted_index]),
        threshold=bundle.manifest.threshold,
    )


def predict_batch(bundle: RuntimeBundle, payloads: list[dict[str, object]]) -> BatchPredictionResult:
    if not payloads:
        raise ValueError("Prediction payload list cannot be empty.")
    frame = pd.DataFrame(payloads)
    transformed = transform_features(frame, bundle.preprocessor)
    transformed_frame = pd.DataFrame(transformed, columns=bundle.manifest.feature_columns)
    predictions: list[PredictionResult] = []
    for row in bundle.model.predict_proba(transformed_frame):
        predicted_index = int(row.argmax())
        category = _resolve_category(bundle, predicted_index)
        predictions.append(
            PredictionResult(
                label="normal" if category == "normal" else "attack",
                category=category,
                score=float(row[predicted_index]),
                threshold=bundle.manifest.threshold,
            )
        )
    return BatchPredictionResult(predictions=predictions)


def describe_runtime(bundle: RuntimeBundle) -> dict[str, object]:
    profile_name = bundle.manifest.metadata.get("profile_name", "unknown")
    return {
        "model_name": bundle.manifest.model_name,
        "profile_name": profile_name,
        "threshold": bundle.manifest.threshold,
        "dataset_path": bundle.manifest.dataset_path,
        "feature_count": len(bundle.manifest.feature_columns),
        "feature_columns": bundle.manifest.feature_columns,
        "label_mapping": bundle.manifest.label_mapping,
    }


def _resolve_category(bundle: RuntimeBundle, predicted_index: int) -> str:
    reverse_mapping = {value: key for key, value in bundle.manifest.label_mapping.items()}
    return reverse_mapping.get(predicted_index, "unknown")
