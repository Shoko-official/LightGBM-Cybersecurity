from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


from ids_project.contracts import NSL_KDD_CATEGORY_MAP, NSL_KDD_COLUMNS
from ids_project.quality import evaluate_release_summary

DEFAULT_CLASS_LABELS = ["normal", "dos", "probe", "r2l", "u2r"]
MODEL_SUMMARY_KEYS = ("default_prod", "default-prod", "u2r_specialist", "u2r-specialist")


@dataclass(frozen=True, slots=True)
class DashboardSources:
    release_summary: dict[str, Any]
    external_report: dict[str, Any]
    manifest: dict[str, Any] | None
    artifact_dir: Path

    @property
    def runtime_available(self) -> bool:
        return runtime_files_available(self.artifact_dir)


def load_json_file(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_optional_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    return load_json_file(target)


def load_dashboard_sources(
    *,
    release_summary_path: str | Path,
    external_report_path: str | Path,
    artifact_dir: str | Path,
) -> DashboardSources:
    artifact_path = Path(artifact_dir)
    manifest_path = artifact_path / "manifest.json"
    return DashboardSources(
        release_summary=load_optional_json(release_summary_path),
        external_report=load_optional_json(external_report_path),
        manifest=load_optional_json(manifest_path) if manifest_path.exists() else None,
        artifact_dir=artifact_path,
    )


def runtime_files_available(artifact_dir: str | Path) -> bool:
    path = Path(artifact_dir)
    return all(
        (path / filename).exists()
        for filename in ("manifest.json", "model.joblib", "preprocessor.joblib")
    )


def selected_model_summary(summary: dict[str, Any]) -> dict[str, Any]:
    for key in MODEL_SUMMARY_KEYS:
        payload = summary.get(key)
        if isinstance(payload, dict):
            return payload
    return summary


def release_status(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    if not summary:
        return False, ["Release summary not found."]
    result = evaluate_release_summary(summary)
    return result.passed, result.failures


def kpi_cards(summary: dict[str, Any]) -> list[dict[str, Any]]:
    model_summary = selected_model_summary(summary)
    metrics = model_summary.get("metrics", {})
    rare_class_f1 = model_summary.get("rare_class_f1", {})
    return [
        _kpi("Accuracy", metrics.get("accuracy")),
        _kpi("Macro F1", metrics.get("f1_score")),
        _kpi("Recall attaque", metrics.get("attack_recall", metrics.get("recall"))),
        _kpi("R2L F1", rare_class_f1.get("r2l")),
        _kpi("U2R F1", rare_class_f1.get("u2r")),
    ]


def class_labels(report: dict[str, Any], manifest: dict[str, Any] | None = None) -> list[str]:
    labels = report.get("class_labels")
    if isinstance(labels, list) and labels:
        return [str(label) for label in labels]
    mapping = (manifest or {}).get("label_mapping")
    if isinstance(mapping, dict) and mapping:
        reverse = {int(index): label for label, index in mapping.items()}
        return [reverse[index] for index in sorted(reverse)]
    return list(DEFAULT_CLASS_LABELS)


def classification_frame(report: dict[str, Any], manifest: dict[str, Any] | None = None) -> pd.DataFrame:
    payload = report.get("classification_report", {})
    labels = class_labels(report, manifest)
    rows = []
    for index, label in enumerate(labels):
        values = payload.get(label) or payload.get(str(index))
        if not isinstance(values, dict):
            continue
        rows.append(
            {
                "Classe": label,
                "Precision": float(values.get("precision", 0.0)),
                "Recall": float(values.get("recall", 0.0)),
                "F1": float(values.get("f1-score", 0.0)),
                "Support": int(float(values.get("support", 0.0))),
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix_frame(report: dict[str, Any], manifest: dict[str, Any] | None = None) -> pd.DataFrame:
    matrix = report.get("confusion_matrix", [])
    labels = class_labels(report, manifest)
    return pd.DataFrame(matrix, index=labels[: len(matrix)], columns=labels[: len(matrix)])


def top_features_frame(report: dict[str, Any], *, limit: int = 12) -> pd.DataFrame:
    rows = report.get("top_features", [])[:limit]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["Feature", "Importance"])
    return frame.rename(columns={"feature": "Feature", "importance": "Importance"})


def support_distribution_frame(report: dict[str, Any], manifest: dict[str, Any] | None = None) -> pd.DataFrame:
    frame = classification_frame(report, manifest)
    if frame.empty:
        return pd.DataFrame(columns=["Classe", "Support"])
    return frame[["Classe", "Support"]]


def runtime_summary(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest:
        return {}
    metadata = manifest.get("metadata", {})
    return {
        "model_name": manifest.get("model_name", "unknown"),
        "profile_name": metadata.get("profile_name", "unknown"),
        "threshold": manifest.get("threshold", "unknown"),
        "dataset_path": manifest.get("dataset_path", "unknown"),
        "feature_count": len(manifest.get("feature_columns", [])),
        "label_mapping": manifest.get("label_mapping", {}),
        "artifact_hashes": metadata.get("artifact_hashes", {}),
        "metadata": metadata,
    }


def simulation_samples(
    manifest: dict[str, Any] | None,
    *,
    preferred_path: str | Path | None = None,
    max_rows: int | None = 1200,
) -> pd.DataFrame:
    dataset_path = Path(preferred_path) if preferred_path else None
    if dataset_path is None and manifest:
        dataset_value = manifest.get("dataset_path")
        if dataset_value:
            dataset_path = Path(dataset_value)
    if dataset_path is None or not dataset_path.exists():
        return pd.DataFrame()

    frame = _read_nsl_dataset(dataset_path)
    if frame.empty or "label" not in frame.columns:
        return pd.DataFrame()

    sample = frame[frame["label"] != "normal"].copy()
    if sample.empty:
        return pd.DataFrame()

    sample["category"] = sample["label"].map(NSL_KDD_CATEGORY_MAP).fillna("unknown")
    columns = ["label", "category", *NSL_KDD_COLUMNS]
    sample = sample[columns]
    if max_rows and len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=42)
    return sample.reset_index(drop=True)


def _kpi(label: str, value: Any) -> dict[str, Any]:
    numeric = None if value is None else float(value)
    return {
        "label": label,
        "value": "N/A" if numeric is None else f"{numeric:.3f}",
        "raw": numeric,
    }


@st.cache_data(show_spinner=False)
def _read_nsl_dataset(path: Path) -> pd.DataFrame:
    separator = _detect_separator(path)
    raw = pd.read_csv(path, sep=separator)
    expected_with_target = len(NSL_KDD_COLUMNS) + 1
    expected_with_difficulty = len(NSL_KDD_COLUMNS) + 2
    if raw.shape[1] in {expected_with_target, expected_with_difficulty} and not set(NSL_KDD_COLUMNS).issubset(
        {str(column).strip().lower() for column in raw.columns}
    ):
        columns = [*NSL_KDD_COLUMNS, "label", "difficulty"]
        raw = pd.read_csv(path, sep=separator, header=None, names=columns)
    normalized = raw.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    if "difficulty" in normalized.columns:
        normalized = normalized.drop(columns=["difficulty"])
    if not set([*NSL_KDD_COLUMNS, "label"]).issubset(normalized.columns):
        return pd.DataFrame()
    normalized["label"] = normalized["label"].astype(str).str.strip()
    for column in NSL_KDD_COLUMNS:
        if column in {"protocol_type", "service", "flag"}:
            normalized[column] = normalized[column].astype(str).str.strip()
        else:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized.dropna(subset=NSL_KDD_COLUMNS)


def _detect_separator(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    preview = path.open("r", encoding="utf-8", errors="ignore").readline()
    if "\t" in preview and preview.count("\t") > preview.count(","):
        return "\t"
    return ","
