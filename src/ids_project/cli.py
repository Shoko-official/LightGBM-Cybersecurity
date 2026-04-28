from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--artifact-dir", default="artifacts/latest")
    parser.add_argument("--report-dir", default="reports/latest")
    parser.add_argument("--profile", default="default-prod")
    parser.add_argument("--estimators", type=int)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--gpu-platform-id", type=int, default=0)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--no-smote", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument(
        "--class-weight",
        action="append",
        default=[],
        metavar="LABEL=WEIGHT",
        help="Repeat to override class weights for the main model.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ids-cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    add_training_arguments(train_parser)

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
        from ids_project.training import train

        result = train(build_training_config(args))
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


def build_training_config(args: argparse.Namespace):
    from ids_project.config import TrainingConfig, build_profile_config

    class_weights = _parse_class_weights(args.class_weight)
    config_kwargs = build_profile_config(args.profile)
    config_kwargs.update(
        dict(
            dataset_path=Path(args.dataset),
            artifact_dir=Path(args.artifact_dir),
            report_dir=Path(args.report_dir),
            profile_name=args.profile,
            use_gpu=args.gpu,
            gpu_platform_id=args.gpu_platform_id,
            gpu_device_id=args.gpu_device_id,
            progress_bar=not args.no_progress,
        )
    )
    if args.estimators is not None:
        config_kwargs["n_estimators"] = args.estimators
    if args.no_smote:
        config_kwargs["use_smote"] = False
    if args.no_class_weights:
        config_kwargs["custom_class_weights"] = None
    if class_weights:
        config_kwargs["custom_class_weights"] = class_weights
    return TrainingConfig(**config_kwargs)


def _parse_class_weights(entries: list[str]) -> dict[str, float]:
    class_weights: dict[str, float] = {}
    for entry in entries:
        label, separator, raw_weight = entry.partition("=")
        if not separator or not label.strip():
            raise ValueError(f"Invalid class weight format: {entry!r}. Expected LABEL=WEIGHT.")
        try:
            class_weights[label.strip()] = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"Invalid class weight value for {label.strip()!r}: {raw_weight!r}.") from exc
    return class_weights


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
