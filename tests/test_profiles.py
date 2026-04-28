from __future__ import annotations

import pytest

from ids_project.config import (
    PRODUCTION_PROFILE_NAME,
    U2R_SPECIALIST_PROFILE_NAME,
    build_profile_config,
)


def test_build_profile_config_returns_production_defaults():
    profile = build_profile_config(PRODUCTION_PROFILE_NAME)

    assert profile["n_estimators"] == 200
    assert profile["custom_class_weights"]["u2r"] == 2.5


def test_build_profile_config_returns_specialist_defaults():
    profile = build_profile_config(U2R_SPECIALIST_PROFILE_NAME)

    assert profile["n_estimators"] == 260
    assert profile["num_leaves"] == 15
    assert profile["max_depth"] == 8


def test_build_profile_config_rejects_unknown_profile():
    with pytest.raises(ValueError, match="Unknown training profile"):
        build_profile_config("unknown-profile")
