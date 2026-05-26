from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_RELEASE_THRESHOLDS = {
    "accuracy": 0.70,
    "macro_f1": 0.55,
    "macro_recall": 0.50,
    "r2l_f1": 0.30,
    "u2r_f1": 0.10,
}


@dataclass(frozen=True, slots=True)
class ReleaseGateResult:
    passed: bool
    failures: list[str]


def evaluate_release_summary(
    summary: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> ReleaseGateResult:
    active_thresholds = thresholds or DEFAULT_RELEASE_THRESHOLDS
    failures: list[str] = []
    evaluated = 0
    for model_key in ("default_prod", "default-prod", "u2r_specialist", "u2r-specialist"):
        payload = summary.get(model_key)
        if payload is None:
            continue
        evaluated += 1
        failures.extend(_evaluate_model_payload(model_key, payload, active_thresholds))
    if evaluated == 0:
        failures.extend(_evaluate_model_payload("summary", summary, active_thresholds))
    return ReleaseGateResult(passed=not failures, failures=failures)


def _evaluate_model_payload(
    name: str,
    payload: dict[str, Any],
    thresholds: dict[str, float],
) -> list[str]:
    metrics = payload.get("metrics", {})
    rare_class_f1 = payload.get("rare_class_f1", {})
    checks = {
        "accuracy": _metric(metrics, "accuracy"),
        "macro_f1": _metric(metrics, "f1_score"),
        "macro_recall": _metric(metrics, "recall"),
        "r2l_f1": _metric(rare_class_f1, "r2l"),
        "u2r_f1": _metric(rare_class_f1, "u2r"),
    }

    failures: list[str] = []
    for key, threshold in thresholds.items():
        value = checks.get(key)
        if value is None:
            failures.append(f"{name}: missing release metric {key}")
        elif value < threshold:
            failures.append(f"{name}: {key}={value:.4f} is below required {threshold:.4f}")
    return failures


def _metric(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    return float(value)
