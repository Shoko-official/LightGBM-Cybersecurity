from __future__ import annotations

import json

from ids_project.quality import evaluate_release_summary


def test_release_gates_accept_current_summary_shape():
    summary = {
        "default_prod": {
            "metrics": {"accuracy": 0.78, "recall": 0.56, "f1_score": 0.58},
            "rare_class_f1": {"r2l": 0.35, "u2r": 0.12},
        },
        "u2r_specialist": {
            "metrics": {"accuracy": 0.77, "recall": 0.54, "f1_score": 0.57},
            "rare_class_f1": {"r2l": 0.32, "u2r": 0.17},
        },
    }

    result = evaluate_release_summary(summary)

    assert result.passed is True
    assert result.failures == []


def test_release_gates_reject_weak_rare_class_scores():
    summary = {
        "default_prod": {
            "metrics": {"accuracy": 0.78, "recall": 0.56, "f1_score": 0.58},
            "rare_class_f1": {"r2l": 0.35, "u2r": 0.01},
        }
    }

    result = evaluate_release_summary(summary)

    assert result.passed is False
    assert any("u2r_f1" in failure for failure in result.failures)


def test_commercial_release_gate_rejects_missing_evidence(tmp_path):
    summary = {
        "release_ready": True,
        "default_prod": {
            "metrics": {"accuracy": 0.78, "recall": 0.56, "f1_score": 0.58},
            "rare_class_f1": {"r2l": 0.35, "u2r": 0.12},
        },
    }

    result = evaluate_release_summary(summary, require_evidence=True, workspace_root=tmp_path)

    assert result.passed is False
    assert any("artifact_dir" in failure for failure in result.failures)
    assert any("report_path" in failure for failure in result.failures)


def test_commercial_release_gate_accepts_complete_evidence(tmp_path):
    dataset_path = tmp_path / "data" / "raw" / "KDDTrain+.txt"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("dataset", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts" / "final"
    artifact_dir.mkdir(parents=True)
    for filename in ("model.joblib", "preprocessor.joblib"):
        (artifact_dir / filename).write_bytes(b"artifact")
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_name": "lightgbm",
                "dataset_path": "data/raw/KDDTrain+.txt",
                "target_column": "label",
                "threshold": 0.5,
                "random_state": 42,
                "feature_columns": ["feature"],
                "categorical_columns": [],
                "numeric_columns": ["feature"],
                "label_mapping": {"normal": 0, "dos": 1},
                "baseline_metrics": {},
                "validation_metrics": {},
                "files": {
                    "model": "model.joblib",
                    "preprocessor": "preprocessor.joblib",
                    "manifest": "manifest.json",
                },
                "metadata": {
                    "artifact_schema_version": 2,
                    "dataset_sha256": "abc",
                    "dependency_versions": {"lightgbm": "4.6.0"},
                },
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "reports" / "external_validation" / "default-prod.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "accuracy": 0.78,
                    "precision": 0.8,
                    "recall": 0.56,
                    "f1_score": 0.58,
                    "roc_auc": 0.9,
                    "attack_precision": 0.85,
                    "attack_recall": 0.82,
                    "attack_f1_score": 0.83,
                    "attack_roc_auc": 0.91,
                    "attack_average_precision": 0.9,
                },
                "classification_report": {"macro avg": {"support": 1200}},
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "release_ready": True,
        "default_prod": {
            "artifact_dir": "artifacts/final",
            "report_path": "reports/external_validation/default-prod.json",
            "metrics": {
                "accuracy": 0.78,
                "precision": 0.8,
                "recall": 0.56,
                "f1_score": 0.58,
                "roc_auc": 0.9,
            },
            "rare_class_f1": {"r2l": 0.35, "u2r": 0.12},
        },
    }

    result = evaluate_release_summary(summary, require_evidence=True, workspace_root=tmp_path)

    assert result.passed is True
    assert result.failures == []
