from __future__ import annotations

import pandas as pd

from ids_project.artifacts import load_runtime_bundle
from ids_project.contracts import BatchPredictionResult, PredictionResult, RuntimeBundle
from ids_project.decision import apply_attack_threshold, model_class_indices
from ids_project.preprocessing import transform_features


def load_runtime(path: str) -> RuntimeBundle:
    return load_runtime_bundle(path)


def _payload_to_frame(payload: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([payload])


def _transform_to_model_frame(bundle: RuntimeBundle, frame: pd.DataFrame) -> pd.DataFrame:
    features = transform_features(frame, bundle.preprocessor)
    return pd.DataFrame(features, columns=bundle.manifest.feature_columns)


def _result_from_decision(
    bundle: RuntimeBundle,
    decision,
    index: int,
) -> PredictionResult:
    reverse_mapping = {v: k for k, v in bundle.manifest.label_mapping.items()}
    normal_index = int(bundle.manifest.label_mapping.get("normal", 0))
    predicted_index = int(decision.predicted_indices[index])
    category = reverse_mapping.get(predicted_index, "unknown")
    label = "normal" if predicted_index == normal_index else "attack"
    score = float(decision.attack_scores[index])
    return PredictionResult(
        label=label,
        category=category,
        score=score,
        threshold=bundle.manifest.threshold,
    )


def predict_one(bundle: RuntimeBundle, payload: dict[str, object]) -> PredictionResult:
    frame = _payload_to_frame(payload)
    features = _transform_to_model_frame(bundle, frame)
    probabilities = bundle.model.predict_proba(features)
    decision = _decide(bundle, probabilities)
    return _result_from_decision(bundle, decision, 0)


def predict_batch(bundle: RuntimeBundle, payloads: list[dict[str, object]]) -> BatchPredictionResult:
    if not payloads:
        raise ValueError("Prediction payload list cannot be empty.")
    frame = pd.DataFrame(payloads)
    features = _transform_to_model_frame(bundle, frame)
    probabilities = bundle.model.predict_proba(features)
    decision = _decide(bundle, probabilities)
    predictions = [
        _result_from_decision(bundle, decision, i) for i in range(len(payloads))
    ]
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
