from __future__ import annotations

from pathlib import Path

import pytest

from interpretability.config import experiment_config, resources_config, scoring_config


def test_config_helpers_apply_defaults() -> None:
    config: dict = {}

    experiment = experiment_config(config)
    resources = resources_config(config)

    assert experiment.trials == 1
    assert experiment.run_root == Path("runs")
    assert experiment.baseline_functionality is None
    assert experiment.functionality_floor_ratio == 0.90
    assert resources.max_cpu_threads == 4
    assert resources.task_timeout_s == 45


def test_scoring_config_rejects_missing_task_paths() -> None:
    with pytest.raises(SystemExit, match="No task_paths configured"):
        scoring_config({"scoring": {"metrics_config": "configs/metrics.yaml"}})


def test_scoring_config_reads_task_paths_and_metrics_default() -> None:
    scoring = scoring_config({"scoring": {"task_paths": ["tasks/replace_token.yaml"]}})

    assert scoring.metrics_config == Path("configs/metrics.yaml")
    assert scoring.task_paths == ["tasks/replace_token.yaml"]


def test_dependency_guidance_uses_single_environment() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    marker = Path("requirements-vllm.txt").read_text(encoding="utf-8")

    assert "same active `.venv`" in readme
    assert "python3 -m venv .venv-vllm" not in readme
    assert "Deprecated optional marker" in marker
