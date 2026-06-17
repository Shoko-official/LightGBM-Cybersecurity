from __future__ import annotations

import numpy as np
import pandas as pd

from ids_project.decision import apply_attack_threshold
from ids_project.evaluation import build_evaluation_report


class FixedProbabilityModel:
    def __init__(self, probabilities):
        self.classes_ = np.array([0, 1, 2])
        self._probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, features):
        return self._probabilities


def test_attack_threshold_controls_normal_vs_attack_decision():
    probabilities = np.array(
        [
            [0.60, 0.35, 0.05],
            [0.40, 0.35, 0.25],
        ]
    )

    strict = apply_attack_threshold(probabilities, np.array([0, 1, 2]), normal_index=0, threshold=0.70)
    sensitive = apply_attack_threshold(probabilities, np.array([0, 1, 2]), normal_index=0, threshold=0.50)

    np.testing.assert_array_equal(strict.predicted_indices, np.array([0, 0]))
    np.testing.assert_array_equal(sensitive.predicted_indices, np.array([0, 1]))


def test_evaluation_reports_multiclass_average_precision():
    model = FixedProbabilityModel(
        [
            [0.90, 0.08, 0.02],
            [0.10, 0.80, 0.10],
            [0.05, 0.10, 0.85],
            [0.80, 0.10, 0.10],
            [0.20, 0.40, 0.40],
        ]
    )
    features = pd.DataFrame({"feature": [0, 1, 2, 3, 4]})
    labels = np.array([0, 1, 2, 0, -1])

    report = build_evaluation_report(
        model=model,
        features=features,
        labels=labels,
        feature_names=["feature"],
        model_name="fixed",
        threshold=0.5,
        split_name="unit",
        top_k_features=1,
        precision_digits=4,
        label_mapping={"normal": 0, "dos": 1, "probe": 2},
    )

    assert report.metrics.average_precision > 0.0
    assert report.metrics.attack_average_precision > 0.0
    assert report.class_labels == ["unknown", "normal", "dos", "probe"]
