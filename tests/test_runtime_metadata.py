from __future__ import annotations

from ids_project.config import PRODUCTION_PROFILE_NAME
from ids_project.runtime import describe_runtime, load_runtime


def test_describe_runtime_exposes_profile_metadata(trained_artifact_dir):
    bundle = load_runtime(trained_artifact_dir)

    summary = describe_runtime(bundle)

    assert summary["profile_name"] == PRODUCTION_PROFILE_NAME
    assert summary["feature_count"] == len(summary["feature_columns"])
