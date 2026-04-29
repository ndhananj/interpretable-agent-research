from __future__ import annotations

from interpretability.resources import ResourceSnapshot, should_backoff


def test_backoff_on_high_load() -> None:
    snapshot = ResourceSnapshot(4, 10.0, 2.5, 16.0, 12.0, False, 0, False, False)
    backoff, reason = should_backoff({"max_load_ratio": 0.75, "min_available_memory_gib": 4}, snapshot)
    assert backoff
    assert "load ratio" in reason


def test_no_backoff_when_resources_available() -> None:
    snapshot = ResourceSnapshot(4, 1.0, 0.25, 16.0, 12.0, False, 0, False, False)
    backoff, reason = should_backoff({"max_load_ratio": 0.75, "min_available_memory_gib": 4}, snapshot)
    assert not backoff
    assert reason == "resources available"

