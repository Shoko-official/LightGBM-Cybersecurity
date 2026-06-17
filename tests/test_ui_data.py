from __future__ import annotations

import json

from ids_project.ui.data import (
    classification_frame,
    confusion_matrix_frame,
    kpi_cards,
    load_dashboard_sources,
    release_status,
    runtime_files_available,
    simulation_samples,
    support_distribution_frame,
    top_features_frame,
)


def test_load_dashboard_sources_accepts_missing_artifacts(tmp_path):
    release_path = tmp_path / "release.json"
    report_path = tmp_path / "external.json"
    release_path.write_text(
        json.dumps(
            {
                "default_prod": {
                    "metrics": {"accuracy": 0.8, "recall": 0.6, "f1_score": 0.7},
                    "rare_class_f1": {"r2l": 0.4, "u2r": 0.2},
                }
            }
        ),
        encoding="utf-8",
    )
    report_path.write_text(json.dumps({"classification_report": {}}), encoding="utf-8")

    sources = load_dashboard_sources(
        release_summary_path=release_path,
        external_report_path=report_path,
        artifact_dir=tmp_path / "missing-artifacts",
    )

    assert sources.release_summary["default_prod"]["metrics"]["accuracy"] == 0.8
    assert sources.runtime_available is False


def test_release_status_uses_quality_gates():
    summary = {
        "default_prod": {
            "metrics": {"accuracy": 0.81, "recall": 0.59, "f1_score": 0.62},
            "rare_class_f1": {"r2l": 0.40, "u2r": 0.22},
        }
    }

    passed, failures = release_status(summary)

    assert passed is True
    assert failures == []


def test_kpi_cards_extract_security_metrics():
    summary = {
        "default_prod": {
            "metrics": {"accuracy": 0.78, "recall": 0.56, "f1_score": 0.58},
            "rare_class_f1": {"r2l": 0.35, "u2r": 0.12},
        }
    }

    cards = kpi_cards(summary)

    assert [card["label"] for card in cards] == [
        "Accuracy",
        "Macro F1",
        "Recall attaque",
        "R2L F1",
        "U2R F1",
    ]
    assert cards[0]["value"] == "0.780"


def test_report_frames_extract_confusion_metrics_and_features():
    report = {
        "confusion_matrix": [[9, 1], [2, 8]],
        "classification_report": {
            "normal": {"precision": 0.82, "recall": 0.9, "f1-score": 0.86, "support": 10},
            "dos": {"precision": 0.89, "recall": 0.8, "f1-score": 0.84, "support": 10},
        },
        "class_labels": ["normal", "dos"],
        "top_features": [{"feature": "src_bytes", "importance": 12.0}],
    }

    metrics = classification_frame(report)
    confusion = confusion_matrix_frame(report)
    features = top_features_frame(report)
    distribution = support_distribution_frame(report)

    assert metrics["Classe"].tolist() == ["normal", "dos"]
    assert confusion.loc["normal", "dos"] == 1
    assert features.iloc[0]["Feature"] == "src_bytes"
    assert distribution["Support"].tolist() == [10, 10]


def test_runtime_files_available_requires_complete_bundle(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    for name in ("manifest.json", "model.joblib"):
        (artifact_dir / name).write_text("", encoding="utf-8")

    assert runtime_files_available(artifact_dir) is False

    (artifact_dir / "preprocessor.joblib").write_text("", encoding="utf-8")
    assert runtime_files_available(artifact_dir) is True


def test_simulation_samples_extracts_attack_rows_from_headerless_dataset(tmp_path):
    dataset_path = tmp_path / "KDDTest+.txt"
    dataset_path.write_text(
        "0,tcp,http,SF,100,200,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,5,5,0.0,0.0,0.0,0.0,1.0,0.0,0.0,10,10,1.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,normal,10\n"
        "0,tcp,private,S0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,30,30,1.0,1.0,0.0,0.0,0.1,0.9,0.0,255,10,0.04,0.8,0.1,0.2,1.0,1.0,0.0,0.0,neptune,10\n",
        encoding="utf-8",
    )

    sample = simulation_samples({"dataset_path": str(dataset_path)}, preferred_path=dataset_path)

    assert len(sample) == 1
    assert sample.iloc[0]["label"] == "neptune"
    assert sample.iloc[0]["category"] == "dos"


def test_simulation_samples_zero_max_rows_keeps_all_attack_rows(tmp_path):
    dataset_path = tmp_path / "KDDTest+.txt"
    dataset_path.write_text(
        "0,tcp,http,SF,100,200,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,5,5,0.0,0.0,0.0,0.0,1.0,0.0,0.0,10,10,1.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,normal,10\n"
        "0,tcp,private,S0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,30,30,1.0,1.0,0.0,0.0,0.1,0.9,0.0,255,10,0.04,0.8,0.1,0.2,1.0,1.0,0.0,0.0,neptune,10\n"
        "0,udp,private,SF,28,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,8,0.0,0.0,0.0,0.0,1.0,0.0,0.0,20,20,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,teardrop,10\n",
        encoding="utf-8",
    )

    sample = simulation_samples({"dataset_path": str(dataset_path)}, preferred_path=dataset_path, max_rows=0)

    assert sample["label"].tolist() == ["neptune", "teardrop"]
