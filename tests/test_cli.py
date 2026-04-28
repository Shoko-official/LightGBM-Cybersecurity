from __future__ import annotations

from ids_project.config import DEFAULT_CLASS_WEIGHTS, PRODUCTION_PROFILE_NAME, U2R_SPECIALIST_PROFILE_NAME
from ids_project.cli import build_parser, build_training_config


def test_cli_parser_accepts_train_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--dataset",
            "data/raw/nsl_kdd.csv",
            "--artifact-dir",
            "artifacts/latest",
            "--report-dir",
            "reports/latest",
            "--estimators",
            "200",
            "--gpu",
            "--no-smote",
            "--no-progress",
            "--class-weight",
            "u2r=20",
            "--class-weight",
            "r2l=10",
        ]
    )

    assert args.command == "train"
    assert args.dataset.endswith("nsl_kdd.csv")
    assert args.estimators == 200
    assert args.gpu is True
    assert args.no_smote is True
    assert args.no_progress is True
    assert args.class_weight == ["u2r=20", "r2l=10"]
    assert args.profile == PRODUCTION_PROFILE_NAME


def test_build_training_config_maps_train_arguments():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--dataset",
            "data/raw/nsl_kdd.csv",
            "--gpu-platform-id",
            "1",
            "--gpu-device-id",
            "2",
            "--class-weight",
            "probe=3.5",
        ]
    )

    config = build_training_config(args)

    assert config.use_gpu is False
    assert config.gpu_platform_id == 1
    assert config.gpu_device_id == 2
    assert config.custom_class_weights == {"probe": 3.5}


def test_build_training_config_uses_default_class_weights():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--dataset",
            "data/raw/nsl_kdd.csv",
        ]
    )

    config = build_training_config(args)

    assert config.profile_name == PRODUCTION_PROFILE_NAME
    assert config.custom_class_weights == DEFAULT_CLASS_WEIGHTS


def test_build_training_config_supports_specialist_profile():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--dataset",
            "data/raw/nsl_kdd.csv",
            "--profile",
            U2R_SPECIALIST_PROFILE_NAME,
        ]
    )

    config = build_training_config(args)

    assert config.profile_name == U2R_SPECIALIST_PROFILE_NAME
    assert config.n_estimators == 260
    assert config.num_leaves == 15
    assert config.max_depth == 8


def test_build_training_config_can_disable_profile_class_weights():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--dataset",
            "data/raw/nsl_kdd.csv",
            "--no-class-weights",
        ]
    )

    config = build_training_config(args)

    assert config.custom_class_weights is None


def test_cli_parser_accepts_predict_one_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "predict-one",
            "--artifact-dir",
            "artifacts/latest",
            "--input",
            "input.json",
        ]
    )

    assert args.command == "predict-one"
    assert args.input == "input.json"


def test_cli_parser_accepts_runtime_inspection():
    parser = build_parser()
    args = parser.parse_args(
        [
            "inspect-runtime",
            "--artifact-dir",
            "artifacts/latest",
        ]
    )

    assert args.command == "inspect-runtime"
    assert args.artifact_dir == "artifacts/latest"
