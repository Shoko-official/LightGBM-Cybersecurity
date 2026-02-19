from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

NUMERIC_COLUMNS = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
 
SKEWED_COLUMNS = [
    "duration",
    "src_bytes",
    "dst_bytes",
]


@dataclass(slots=True)
class DatasetSummary:
    row_count: int
    feature_count: int
    label_distribution: dict[str, int]
    source_path: str


@dataclass(slots=True)
class SplitData:
    features_train: Any
    features_validation: Any
    features_test: Any
    labels_train: Any
    labels_validation: Any
    labels_test: Any
    summary: DatasetSummary


@dataclass(slots=True)
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    average_precision: float


@dataclass(slots=True)
class PredictionResult:
    label: str
    category: str
    score: float
    threshold: float


@dataclass(slots=True)
class BatchPredictionResult:
    predictions: list[PredictionResult]


@dataclass(slots=True)
class EvaluationReport:
    model_name: str
    threshold: float
    metrics: ModelMetrics
    confusion_matrix: list[list[int]]
    classification_report: dict[str, dict[str, float]]
    top_features: list[dict[str, float]]
    split_name: str
    attack_category_recall: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainingResult:
    model_name: str
    artifact_dir: Path
    report_path: Path
    manifest_path: Path
    threshold: float
    baseline_metrics: dict[str, ModelMetrics]
    validation_report: EvaluationReport


@dataclass(slots=True)
class ArtifactManifest:
    model_name: str
    dataset_path: str
    target_column: str
    threshold: float
    random_state: int
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    label_mapping: dict[str, int]
    baseline_metrics: dict[str, dict[str, float]]
    validation_metrics: dict[str, float]
    files: dict[str, str]
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RuntimeBundle:
    preprocessor: Any
    model: Any
    manifest: ArtifactManifest


NSL_KDD_CATEGORY_MAP = {
    "normal": "normal",
    "neptune": "dos",
    "back": "dos",
    "land": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "apache2": "dos",
    "udpstorm": "dos",
    "processtable": "dos",
    "mailbomb": "dos",
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "satan": "probe",
    "mscan": "probe",
    "saint": "probe",
    "ftp_write": "r2l",
    "guess_passwd": "r2l",
    "imap": "r2l",
    "multihop": "r2l",
    "phf": "r2l",
    "spy": "r2l",
    "warezclient": "r2l",
    "warezmaster": "r2l",
    "sendmail": "r2l",
    "named": "r2l",
    "snmpgetattack": "r2l",
    "snmpguess": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",
    "worm": "r2l",
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "perl": "u2r",
    "rootkit": "u2r",
    "httptunnel": "u2r",
    "ps": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
}
