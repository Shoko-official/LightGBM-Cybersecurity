from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def build_baseline_classifier() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        solver="saga",
        max_iter=3000,
        random_state=42,
    )
