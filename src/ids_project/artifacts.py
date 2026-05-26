from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib

from ids_project.contracts import ArtifactManifest, RuntimeBundle
from ids_project.preprocessing import PreprocessingArtifacts


def ensure_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_runtime_bundle(
    artifact_dir: Path,
    preprocessor: PreprocessingArtifacts,
    model: Any,
    manifest: ArtifactManifest,
) -> tuple[Path, Path, Path]:
    target_dir = ensure_directory(artifact_dir)
    preprocessor_path = target_dir / "preprocessor.joblib"
    model_path = target_dir / "model.joblib"
    manifest_path = target_dir / "manifest.json"

    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(model, model_path)
    manifest.metadata["artifact_hashes"] = {
        "preprocessor": _sha256_file(preprocessor_path),
        "model": _sha256_file(model_path),
    }
    save_json(manifest_path, manifest.to_dict())
    return preprocessor_path, model_path, manifest_path


def load_runtime_bundle(artifact_dir: str | Path) -> RuntimeBundle:
    target_dir = Path(artifact_dir).expanduser().resolve()
    manifest_path = target_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Required artifact file not found: {manifest_path}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ArtifactManifest(**manifest_data)
    preprocessor_path = _manifest_file_path(
        target_dir,
        manifest,
        "preprocessor",
        "preprocessor.joblib",
    )
    model_path = _manifest_file_path(target_dir, manifest, "model", "model.joblib")

    for path in (preprocessor_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact file not found: {path}")

    _verify_artifact_hashes(manifest, {"preprocessor": preprocessor_path, "model": model_path})

    return RuntimeBundle(
        preprocessor=joblib.load(preprocessor_path),
        model=joblib.load(model_path),
        manifest=manifest,
    )


def _manifest_file_path(target_dir: Path, manifest: ArtifactManifest, key: str, default: str) -> Path:
    raw_path = manifest.files.get(key, default)
    path = (target_dir / raw_path).resolve()
    if not path.is_relative_to(target_dir):
        raise ValueError(f"Manifest file path for {key!r} escapes artifact directory: {raw_path!r}.")
    return path


def _verify_artifact_hashes(manifest: ArtifactManifest, paths: dict[str, Path]) -> None:
    hashes = manifest.metadata.get("artifact_hashes")
    if not isinstance(hashes, dict):
        return
    for key, path in paths.items():
        expected = hashes.get(key)
        if not expected:
            continue
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(f"Artifact hash mismatch for {key}: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
