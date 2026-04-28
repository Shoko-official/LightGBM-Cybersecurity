from __future__ import annotations

from ids_project.cli import build_parser


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
        ]
    )

    assert args.command == "train"
    assert args.dataset.endswith("nsl_kdd.csv")


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
