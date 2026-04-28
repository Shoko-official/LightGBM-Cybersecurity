from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ids_project.artifacts import ensure_directory, load_runtime_bundle, save_json
from ids_project.config import TrainingConfig
from ids_project.data.dataset import load_dataset
from ids_project.evaluation import evaluate, save_report
from ids_project.training import train


def default_search_candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "baseline_prod",
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 20,
                "feature_fraction": 1.0,
                "bagging_fraction": 1.0,
                "bagging_freq": 0,
                "lambda_l1": 0.0,
                "lambda_l2": 0.0,
                "use_smote": True,
            },
        },
        {
            "name": "regularized_bagging",
            "params": {
                "n_estimators": 240,
                "learning_rate": 0.04,
                "num_leaves": 31,
                "min_child_samples": 40,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "lambda_l1": 0.0,
                "lambda_l2": 1.0,
                "use_smote": True,
            },
        },
        {
            "name": "wider_trees",
            "params": {
                "n_estimators": 220,
                "learning_rate": 0.04,
                "num_leaves": 63,
                "min_child_samples": 20,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "bagging_freq": 1,
                "lambda_l1": 0.0,
                "lambda_l2": 0.5,
                "use_smote": True,
            },
        },
        {
            "name": "shallow_regularized",
            "params": {
                "n_estimators": 260,
                "learning_rate": 0.06,
                "num_leaves": 15,
                "max_depth": 8,
                "min_child_samples": 40,
                "feature_fraction": 0.85,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "lambda_l1": 0.2,
                "lambda_l2": 1.0,
                "use_smote": True,
            },
        },
        {
            "name": "rare_class_focus",
            "params": {
                "n_estimators": 240,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "min_child_samples": 10,
                "feature_fraction": 0.95,
                "bagging_fraction": 0.9,
                "bagging_freq": 1,
                "lambda_l1": 0.0,
                "lambda_l2": 0.2,
                "use_smote": True,
            },
        },
    ]


def run_model_search(
    *,
    train_dataset: Path,
    external_dataset: Path,
    artifact_root: Path,
    report_root: Path,
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidate_specs = candidates or default_search_candidates()
    external_config = TrainingConfig(dataset_path=external_dataset)
    external_frame = load_dataset(external_config)
    external_features = external_frame.drop(columns=["label"])
    external_labels = external_frame["label"]

    results: list[dict[str, Any]] = []
    target_artifact_root = ensure_directory(artifact_root)
    target_report_root = ensure_directory(report_root)
    workspace_root = Path.cwd().resolve()

    for index, candidate in enumerate(candidate_specs, start=1):
        name = candidate["name"]
        params = dict(candidate["params"])
        artifact_dir = target_artifact_root / name
        candidate_report_dir = target_report_root / name
        config = TrainingConfig(
            dataset_path=train_dataset,
            artifact_dir=artifact_dir,
            report_dir=candidate_report_dir,
            progress_bar=False,
            notes=f"model_search:{name}",
            extra_metadata={"search_candidate": name, "search_index": index},
            **params,
        )
        training_result = train(config)
        bundle = load_runtime_bundle(training_result.artifact_dir)
        external_report = evaluate(bundle, (external_features, external_labels), split_name="external")
        external_report_path = save_report(external_report, candidate_report_dir, "external_report.json")
        validation_metrics = training_result.validation_report.metrics
        external_metrics = external_report.metrics
        validation_classification = training_result.validation_report.classification_report
        external_classification = external_report.classification_report
        result = {
            "name": name,
            "params": params,
            "artifact_dir": _relative_path(training_result.artifact_dir, workspace_root),
            "report_dir": _relative_path(candidate_report_dir, workspace_root),
            "validation_metrics": {
                "accuracy": validation_metrics.accuracy,
                "macro_precision": validation_metrics.precision,
                "macro_recall": validation_metrics.recall,
                "macro_f1": validation_metrics.f1_score,
                "roc_auc": validation_metrics.roc_auc,
            },
            "external_metrics": {
                "accuracy": external_metrics.accuracy,
                "macro_precision": external_metrics.precision,
                "macro_recall": external_metrics.recall,
                "macro_f1": external_metrics.f1_score,
                "roc_auc": external_metrics.roc_auc,
            },
            "rare_class_f1": {
                "validation_r2l": float(validation_classification.get("3", {}).get("f1-score", 0.0)),
                "validation_u2r": float(validation_classification.get("4", {}).get("f1-score", 0.0)),
                "external_r2l": float(external_classification.get("3", {}).get("f1-score", 0.0)),
                "external_u2r": float(external_classification.get("4", {}).get("f1-score", 0.0)),
            },
            "validation_report_path": _relative_path(training_result.report_path, workspace_root),
            "external_report_path": _relative_path(external_report_path, workspace_root),
        }
        results.append(result)

    ranked_results = sorted(results, key=_ranking_key, reverse=True)
    best_result = ranked_results[0]
    best_alias_dir = target_artifact_root / "best"
    if best_alias_dir.exists():
        shutil.rmtree(best_alias_dir)
    shutil.copytree(workspace_root / best_result["artifact_dir"], best_alias_dir)

    summary = {
        "train_dataset": _relative_path(train_dataset, workspace_root),
        "external_dataset": _relative_path(external_dataset, workspace_root),
        "candidate_count": len(results),
        "best_candidate": best_result["name"],
        "best_artifact_dir": _relative_path(best_alias_dir, workspace_root),
        "production_defaults": best_result["params"],
        "ranked_results": ranked_results,
    }
    save_json(target_report_root / "leaderboard.json", summary)
    save_json(target_report_root / "best_candidate.json", best_result)
    (target_report_root / "leaderboard.md").write_text(_build_markdown_leaderboard(summary), encoding="utf-8")
    return summary


def _ranking_key(result: dict[str, Any]) -> tuple[float, float, float, float, float]:
    external_metrics = result["external_metrics"]
    validation_metrics = result["validation_metrics"]
    r2l_f1 = float(result["rare_class_f1"]["external_r2l"])
    u2r_f1 = float(result["rare_class_f1"]["external_u2r"])
    return (
        float(external_metrics["macro_f1"]),
        u2r_f1,
        r2l_f1,
        float(validation_metrics["macro_f1"]),
        float(external_metrics["accuracy"]),
    )


def _build_markdown_leaderboard(summary: dict[str, Any]) -> str:
    lines = [
        "# Model Search Leaderboard",
        "",
        f"- Train dataset: `{summary['train_dataset']}`",
        f"- External dataset: `{summary['external_dataset']}`",
        f"- Best candidate: `{summary['best_candidate']}`",
        f"- Best artifact directory: `{summary['best_artifact_dir']}`",
        "",
        "| Rank | Candidate | External Macro F1 | External U2R F1 | External R2L F1 | Validation Macro F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for index, result in enumerate(summary["ranked_results"], start=1):
        external_metrics = result["external_metrics"]
        validation_metrics = result["validation_metrics"]
        rare_class_f1 = result["rare_class_f1"]
        lines.append(
            "| "
            f"{index} | {result['name']} | "
            f"{external_metrics['macro_f1']:.4f} | "
            f"{rare_class_f1['external_u2r']:.4f} | "
            f"{rare_class_f1['external_r2l']:.4f} | "
            f"{validation_metrics['macro_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Selected parameters",
            "",
        ]
    )
    for result in summary["ranked_results"]:
        lines.append(f"### {result['name']}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(result["params"], indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _relative_path(path: str | Path, workspace_root: Path) -> str:
    target = Path(path).resolve()
    try:
        return str(target.relative_to(workspace_root))
    except ValueError:
        return str(target)
