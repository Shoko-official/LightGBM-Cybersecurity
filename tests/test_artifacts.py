from __future__ import annotations

from ids_project.artifacts import load_runtime_bundle


def test_manifest_is_written_with_runtime_files(trained_artifact_dir):
    bundle = load_runtime_bundle(trained_artifact_dir)
    manifest = bundle.manifest.to_dict()

    assert bundle.manifest.model_name in {"lightgbm", "logistic_regression"}
    assert manifest["files"]["model"] == "model.joblib"
