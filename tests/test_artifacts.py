from __future__ import annotations

from ids_project.artifacts import load_runtime_bundle


def test_manifest_is_written_with_runtime_files(trained_artifact_dir):
    bundle = load_runtime_bundle(trained_artifact_dir)
    manifest = bundle.manifest.to_dict()

    assert bundle.manifest.model_name in {"lightgbm", "lightgbm_fallback_logistic_regression"}
    assert manifest["files"]["model"] == "model.joblib"
    assert manifest["metadata"]["artifact_hashes"]["model"]


def test_runtime_load_rejects_tampered_artifact(trained_artifact_dir):
    model_path = trained_artifact_dir / "model.joblib"
    with model_path.open("ab") as handle:
        handle.write(b"tampered")

    try:
        try:
            load_runtime_bundle(trained_artifact_dir)
        except ValueError as exc:
            assert "hash mismatch" in str(exc)
        else:
            raise AssertionError("Expected tampered artifact to be rejected.")
    finally:
        model_path.write_bytes(model_path.read_bytes()[:-8])
