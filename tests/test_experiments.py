from __future__ import annotations

from ids_project.experiments import default_search_candidates


def test_default_search_candidates_are_named_and_non_empty():
    candidates = default_search_candidates()

    assert len(candidates) >= 3
    assert all(candidate["name"] for candidate in candidates)
    assert all(isinstance(candidate["params"], dict) for candidate in candidates)
