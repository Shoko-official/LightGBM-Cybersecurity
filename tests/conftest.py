from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ids_project.config import TrainingConfig
from ids_project.training import train


@pytest.fixture()
def sample_dataset_frame() -> pd.DataFrame:
    rows = []
    services = ["http", "smtp", "ftp"]
    flags = ["SF", "S0", "REJ"]
    labels = [
        "normal",
        "neptune",
        "satan",
        "guess_passwd",
        "buffer_overflow",
    ]
    for index in range(60):
        label = labels[index // 12]
        rows.append(
            {
                "duration": index,
                "protocol_type": "tcp" if index % 2 == 0 else "udp",
                "service": services[index % len(services)],
                "flag": flags[index % len(flags)],
                "src_bytes": 100 + index * 5,
                "dst_bytes": 40 + index * 3,
                "land": 0,
                "wrong_fragment": 0,
                "urgent": 0,
                "hot": index % 4,
                "num_failed_logins": 0 if index % 5 else 1,
                "logged_in": 1 if index % 3 else 0,
                "num_compromised": index % 2,
                "root_shell": 0,
                "su_attempted": 0,
                "num_root": index % 3,
                "num_file_creations": index % 2,
                "num_shells": 0,
                "num_access_files": index % 3,
                "num_outbound_cmds": 0,
                "is_host_login": 0,
                "is_guest_login": 0 if index % 4 else 1,
                "count": 10 + index,
                "srv_count": 5 + index,
                "serror_rate": 0.0 if index % 2 else 0.2,
                "srv_serror_rate": 0.0 if index % 2 else 0.2,
                "rerror_rate": 0.1 if index % 3 == 0 else 0.0,
                "srv_rerror_rate": 0.1 if index % 3 == 0 else 0.0,
                "same_srv_rate": 0.7,
                "diff_srv_rate": 0.2,
                "srv_diff_host_rate": 0.3,
                "dst_host_count": 100 + index,
                "dst_host_srv_count": 90 + index,
                "dst_host_same_srv_rate": 0.6,
                "dst_host_diff_srv_rate": 0.1,
                "dst_host_same_src_port_rate": 0.4,
                "dst_host_srv_diff_host_rate": 0.2,
                "dst_host_serror_rate": 0.0 if index % 2 else 0.3,
                "dst_host_srv_serror_rate": 0.0 if index % 2 else 0.3,
                "dst_host_rerror_rate": 0.0 if index % 3 else 0.2,
                "dst_host_srv_rerror_rate": 0.0 if index % 3 else 0.2,
                "label": label,
                "difficulty": index % 5,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_dataset_path(tmp_path: Path, sample_dataset_frame: pd.DataFrame) -> Path:
    path = tmp_path / "nsl_kdd_sample.csv"
    sample_dataset_frame.to_csv(path, index=False)
    return path


@pytest.fixture()
def trained_artifact_dir(tmp_path: Path, sample_dataset_path: Path) -> Path:
    artifact_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "reports"
    train(
        TrainingConfig(
            dataset_path=sample_dataset_path,
            artifact_dir=artifact_dir,
            report_dir=report_dir,
            n_estimators=40,
            early_stopping_rounds=5,
        )
    )
    return artifact_dir
