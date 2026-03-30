from __future__ import annotations

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
    save_json(manifest_path, manifest.to_dict())
    return preprocessor_path, model_path, manifest_path


def load_runtime_bundle(artifact_dir: str | Path) -> RuntimeBundle:
    target_dir = Path(artifact_dir).expanduser().resolve()
    preprocessor_path = target_dir / "preprocessor.joblib"
    model_path = target_dir / "model.joblib"
    manifest_path = target_dir / "manifest.json"

    for path in (preprocessor_path, model_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact file not found: {path}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ArtifactManifest(**manifest_data)

    return RuntimeBundle(
        preprocessor=joblib.load(preprocessor_path),
        model=joblib.load(model_path),
        manifest=manifest,
    )
