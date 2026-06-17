from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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
    *,
    require_evidence: bool = False,
    workspace_root: str | Path | None = None,
    min_external_support: int = 1000,
) -> ReleaseGateResult:
    active_thresholds = thresholds or DEFAULT_RELEASE_THRESHOLDS
    failures: list[str] = []
    root = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
    evaluated = 0
    for model_key in ("default_prod", "default-prod", "u2r_specialist", "u2r-specialist"):
        payload = summary.get(model_key)
        if payload is None:
            continue
        evaluated += 1
        failures.extend(_evaluate_model_payload(model_key, payload, active_thresholds))
        if require_evidence:
            failures.extend(_evaluate_model_evidence(model_key, payload, root, min_external_support))
    if evaluated == 0:
        failures.extend(_evaluate_model_payload("summary", summary, active_thresholds))
        if require_evidence:
            failures.extend(_evaluate_model_evidence("summary", summary, root, min_external_support))
    if require_evidence and summary.get("release_ready") is not True:
        failures.append("release_ready must be true for commercial release evidence")
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


def _evaluate_model_evidence(
    name: str,
    payload: dict[str, Any],
    workspace_root: Path,
    min_external_support: int,
) -> list[str]:
    failures: list[str] = []
    artifact_dir, artifact_dir_failure = _evidence_path(
        name,
        "artifact_dir",
        payload.get("artifact_dir"),
        workspace_root,
    )
    report_path, report_path_failure = _evidence_path(
        name,
        "report_path",
        payload.get("report_path"),
        workspace_root,
    )

    if artifact_dir_failure:
        failures.append(artifact_dir_failure)
    elif not artifact_dir.exists():
        failures.append(f"{name}: artifact_dir does not exist: {artifact_dir}")
    elif not artifact_dir.is_dir():
        failures.append(f"{name}: artifact_dir is not a directory: {artifact_dir}")
    else:
        failures.extend(_evaluate_artifact_manifest(name, artifact_dir, workspace_root))

    if report_path_failure:
        failures.append(report_path_failure)
    elif not report_path.exists():
        failures.append(f"{name}: report_path does not exist: {report_path}")
    elif not report_path.is_file():
        failures.append(f"{name}: report_path is not a file: {report_path}")
    else:
        failures.extend(_evaluate_external_report(name, report_path, payload, min_external_support))

    return failures


def _evidence_path(
    name: str,
    field: str,
    raw_path: Any,
    workspace_root: Path,
) -> tuple[Path | None, str | None]:
    if raw_path is None:
        return None, f"{name}: missing {field}"
    path = Path(str(raw_path)).expanduser()
    resolved = path.resolve() if path.is_absolute() else (workspace_root / path).resolve()
    if not resolved.is_relative_to(workspace_root):
        return None, f"{name}: {field} must be inside workspace: {resolved}"
    return resolved, None


def _workspace_path(raw_path: Any, workspace_root: Path) -> Path | None:
    if raw_path is None:
        return None
    path = Path(str(raw_path)).expanduser()
    resolved = path.resolve() if path.is_absolute() else (workspace_root / path).resolve()
    if not resolved.is_relative_to(workspace_root):
        return None
    return resolved


def _evaluate_artifact_manifest(name: str, artifact_dir: Path, workspace_root: Path) -> list[str]:
    failures: list[str] = []
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        return [f"{name}: manifest.json is missing from artifact_dir"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest.get("metadata", {})
    if manifest.get("model_name") != "lightgbm":
        failures.append(f"{name}: artifact model_name must be lightgbm")
    if metadata.get("artifact_schema_version") != 2:
        failures.append(f"{name}: artifact_schema_version must be 2")
    if not metadata.get("dataset_sha256"):
        failures.append(f"{name}: dataset_sha256 is missing from artifact metadata")
    if not isinstance(metadata.get("dependency_versions"), dict):
        failures.append(f"{name}: dependency_versions are missing from artifact metadata")

    required_files = ("model", "preprocessor", "manifest")
    manifest_files = manifest.get("files", {})
    for key in required_files:
        if not manifest_files.get(key):
            failures.append(f"{name}: manifest file entry missing for {key}")
    for filename in ("model.joblib", "preprocessor.joblib"):
        if not (artifact_dir / filename).exists():
            failures.append(f"{name}: runtime file missing: {filename}")

    dataset_path = _workspace_path(manifest.get("dataset_path"), workspace_root)
    if dataset_path is None or not dataset_path.exists():
        failures.append(f"{name}: manifest dataset_path is missing or outside workspace")
    return failures


def _evaluate_external_report(
    name: str,
    report_path: Path,
    summary_payload: dict[str, Any],
    min_external_support: int,
) -> list[str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    report_metrics = report.get("metrics", {})
    summary_metrics = summary_payload.get("metrics", {})
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        summary_value = summary_metrics.get(key)
        report_value = report_metrics.get(key)
        if summary_value is None or report_value is None:
            failures.append(f"{name}: missing {key} in release summary or report")
            continue
        if abs(float(summary_value) - float(report_value)) > 1e-6:
            failures.append(f"{name}: release summary {key} does not match report")

    support = _report_support(report)
    if support < min_external_support:
        failures.append(
            f"{name}: external support {support} is below required {min_external_support}"
        )
    for key in ("attack_precision", "attack_recall", "attack_f1_score", "attack_roc_auc", "attack_average_precision"):
        if key not in report_metrics:
            failures.append(f"{name}: missing security metric {key}")
    return failures


def _report_support(report: dict[str, Any]) -> int:
    classification = report.get("classification_report", {})
    macro_avg = classification.get("macro avg", {})
    support = macro_avg.get("support")
    if support is None:
        return 0
    return int(float(support))
