from __future__ import annotations

from pathlib import Path

import pytest

from ids_project.config import TrainingConfig
from ids_project.data.dataset import load_dataset
from ids_project.evaluation import evaluate
from ids_project.runtime import load_runtime


@pytest.mark.external
def test_external_validation_runs_when_artifacts_exist():
    base_dir = Path.cwd()
    artifact_dir = base_dir / "artifacts" / "final"
    if not artifact_dir.exists():
        pytest.skip("Production artifacts are not available locally.")

    dataset = load_dataset(TrainingConfig(dataset_path=base_dir / "data" / "raw" / "KDDTest+.txt"))
    bundle = load_runtime(artifact_dir)
    report = evaluate(bundle, (dataset.drop(columns=["label"]), dataset["label"]), split_name="external")

    assert report.metrics.accuracy > 0.7
