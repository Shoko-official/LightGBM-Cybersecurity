from __future__ import annotations

import pytest

from ids_project.contracts import NSL_KDD_COLUMNS
from ids_project.ui.forms import DEFAULT_PAYLOAD, build_payload, validate_payload


def test_build_payload_returns_all_nsl_kdd_columns():
    payload = build_payload({"protocol_type": "udp", "src_bytes": 2048})

    assert list(payload) == NSL_KDD_COLUMNS
    assert payload["protocol_type"] == "udp"
    assert payload["src_bytes"] == 2048


def test_build_payload_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown payload fields"):
        build_payload({"unexpected": 1})


def test_validate_payload_rejects_missing_fields():
    payload = dict(DEFAULT_PAYLOAD)
    payload.pop("duration")

    with pytest.raises(ValueError, match="missing fields"):
        validate_payload(payload)


def test_validate_payload_rejects_non_numeric_values():
    payload = dict(DEFAULT_PAYLOAD)
    payload["src_bytes"] = "not-a-number"

    with pytest.raises(ValueError, match="must be numeric"):
        validate_payload(payload)
