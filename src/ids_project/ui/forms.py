from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ids_project.contracts import CATEGORICAL_COLUMNS, NSL_KDD_COLUMNS, NUMERIC_COLUMNS

PROTOCOL_OPTIONS = ["tcp", "udp", "icmp"]
SERVICE_OPTIONS = [
    "http",
    "private",
    "domain_u",
    "smtp",
    "ftp_data",
    "eco_i",
    "other",
    "ecr_i",
    "telnet",
]
FLAG_OPTIONS = ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "S3", "OTH"]

IMPORTANT_NUMERIC_FIELDS = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "count",
    "srv_count",
    "same_srv_rate",
    "diff_srv_rate",
    "serror_rate",
    "rerror_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "logged_in",
    "is_guest_login",
    "root_shell",
]

DEFAULT_PAYLOAD: dict[str, object] = {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 181,
    "dst_bytes": 5450,
    "land": 0,
    "wrong_fragment": 0,
    "urgent": 0,
    "hot": 0,
    "num_failed_logins": 0,
    "logged_in": 1,
    "num_compromised": 0,
    "root_shell": 0,
    "su_attempted": 0,
    "num_root": 0,
    "num_file_creations": 0,
    "num_shells": 0,
    "num_access_files": 0,
    "num_outbound_cmds": 0,
    "is_host_login": 0,
    "is_guest_login": 0,
    "count": 8,
    "srv_count": 8,
    "serror_rate": 0.0,
    "srv_serror_rate": 0.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0,
    "diff_srv_rate": 0.0,
    "srv_diff_host_rate": 0.0,
    "dst_host_count": 9,
    "dst_host_srv_count": 9,
    "dst_host_same_srv_rate": 1.0,
    "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 0.11,
    "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0,
}

ADVANCED_FIELDS = [
    column
    for column in NUMERIC_COLUMNS
    if column not in set(IMPORTANT_NUMERIC_FIELDS)
]


def build_payload(values: Mapping[str, Any]) -> dict[str, object]:
    unknown = sorted(set(values) - set(NSL_KDD_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown payload fields: {', '.join(unknown)}")

    payload = dict(DEFAULT_PAYLOAD)
    payload.update(values)
    validate_payload(payload)
    return {column: payload[column] for column in NSL_KDD_COLUMNS}


def validate_payload(payload: Mapping[str, Any]) -> None:
    missing = sorted(set(NSL_KDD_COLUMNS) - set(payload))
    extra = sorted(set(payload) - set(NSL_KDD_COLUMNS))
    if missing:
        raise ValueError(f"Prediction payload is missing fields: {', '.join(missing)}")
    if extra:
        raise ValueError(f"Prediction payload contains unexpected fields: {', '.join(extra)}")

    for column in CATEGORICAL_COLUMNS:
        if not str(payload[column]).strip():
            raise ValueError(f"Prediction payload field {column!r} cannot be empty.")

    for column in NUMERIC_COLUMNS:
        try:
            float(payload[column])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Prediction payload field {column!r} must be numeric.") from exc
