from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class DecisionResult:
    predicted_indices: np.ndarray
    attack_scores: np.ndarray
    class_confidences: np.ndarray


def model_class_indices(model: Any, probability_count: int) -> np.ndarray:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return np.arange(probability_count, dtype=int)
    return np.asarray(classes, dtype=int)


def apply_attack_threshold(
    probabilities: np.ndarray,
    class_indices: np.ndarray,
    *,
    normal_index: int,
    threshold: float,
) -> DecisionResult:
    if probabilities.ndim != 2:
        raise ValueError("Prediction probabilities must be a 2D matrix.")
    if probabilities.shape[1] != len(class_indices):
        raise ValueError("Probability columns do not match model classes.")

    normal_positions = np.flatnonzero(class_indices == normal_index)
    if len(normal_positions) == 0:
        predicted_positions = np.argmax(probabilities, axis=1)
        predicted_indices = class_indices[predicted_positions]
        confidences = probabilities[np.arange(len(probabilities)), predicted_positions]
        return DecisionResult(
            predicted_indices=predicted_indices.astype(int),
            attack_scores=np.ones(len(probabilities), dtype=float),
            class_confidences=confidences.astype(float),
        )

    normal_position = int(normal_positions[0])
    normal_scores = probabilities[:, normal_position]
    attack_scores = 1.0 - normal_scores
    attack_positions = np.flatnonzero(class_indices != normal_index)

    predicted_positions = np.full(len(probabilities), normal_position, dtype=int)
    if len(attack_positions) > 0:
        best_attack_offsets = np.argmax(probabilities[:, attack_positions], axis=1)
        best_attack_positions = attack_positions[best_attack_offsets]
        predicted_positions = np.where(
            attack_scores >= threshold,
            best_attack_positions,
            predicted_positions,
        )

    predicted_indices = class_indices[predicted_positions]
    confidences = probabilities[np.arange(len(probabilities)), predicted_positions]
    return DecisionResult(
        predicted_indices=predicted_indices.astype(int),
        attack_scores=attack_scores.astype(float),
        class_confidences=confidences.astype(float),
    )
