from __future__ import annotations

from ids_project.quality import evaluate_release_summary


def test_release_gates_accept_current_summary_shape():
    summary = {
        "default_prod": {
            "metrics": {"accuracy": 0.78, "recall": 0.56, "f1_score": 0.58},
            "rare_class_f1": {"r2l": 0.35, "u2r": 0.12},
        },
        "u2r_specialist": {
            "metrics": {"accuracy": 0.77, "recall": 0.54, "f1_score": 0.57},
            "rare_class_f1": {"r2l": 0.32, "u2r": 0.17},
        },
    }

    result = evaluate_release_summary(summary)

    assert result.passed is True
    assert result.failures == []


def test_release_gates_reject_weak_rare_class_scores():
    summary = {
        "default_prod": {
            "metrics": {"accuracy": 0.78, "recall": 0.56, "f1_score": 0.58},
            "rare_class_f1": {"r2l": 0.35, "u2r": 0.01},
        }
    }

    result = evaluate_release_summary(summary)

    assert result.passed is False
    assert any("u2r_f1" in failure for failure in result.failures)
