from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ids-cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--artifact-dir", default="artifacts/latest")
    train_parser.add_argument("--report-dir", default="reports/latest")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--artifact-dir", required=True)
    eval_parser.add_argument("--dataset", required=True)

    predict_one_parser = subparsers.add_parser("predict-one")
    predict_one_parser.add_argument("--artifact-dir", required=True)
    predict_one_parser.add_argument("--input", required=True)

    predict_batch_parser = subparsers.add_parser("predict-batch")
    predict_batch_parser.add_argument("--artifact-dir", required=True)
    predict_batch_parser.add_argument("--input", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        from ids_project.config import TrainingConfig
        from ids_project.training import train

        result = train(
            TrainingConfig(
                dataset_path=Path(args.dataset),
                artifact_dir=Path(args.artifact_dir),
                report_dir=Path(args.report_dir),
            )
        )
        print(json.dumps(_training_result_to_dict(result), indent=2))
        return

    if args.command == "evaluate":
        from ids_project.config import TrainingConfig
        from ids_project.data.dataset import load_dataset
        from ids_project.evaluation import evaluate
        from ids_project.runtime import load_runtime

        bundle = load_runtime(args.artifact_dir)
        dataset = load_dataset(TrainingConfig(dataset_path=Path(args.dataset)))
        features = dataset.drop(columns=["label"])
        labels = dataset["label"]
        report = evaluate(bundle, (features, labels), split_name="external")
        print(json.dumps(report.to_dict(), indent=2))
        return

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    from ids_project.runtime import load_runtime, predict_batch, predict_one

    bundle = load_runtime(args.artifact_dir)
    if args.command == "predict-one":
        result = predict_one(bundle, payload)
        print(json.dumps(asdict(result), indent=2))
        return

    result = predict_batch(bundle, payload)
    print(json.dumps(asdict(result), indent=2))


def _training_result_to_dict(result) -> dict[str, object]:
    return {
        "model_name": result.model_name,
        "artifact_dir": str(result.artifact_dir),
        "report_path": str(result.report_path),
        "manifest_path": str(result.manifest_path),
        "threshold": result.threshold,
        "baseline_metrics": {name: asdict(metrics) for name, metrics in result.baseline_metrics.items()},
        "validation_report": result.validation_report.to_dict(),
    }


if __name__ == "__main__":
    main()
