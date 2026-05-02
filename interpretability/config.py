from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


@dataclass(frozen=True)
class ExperimentConfig:
    trials: int
    run_root: Path
    baseline_functionality: float | None
    functionality_floor_ratio: float
    daemon_sleep_s: float


@dataclass(frozen=True)
class ResourcesConfig:
    max_cpu_threads: int
    task_timeout_s: int
    values: dict[str, Any]


@dataclass(frozen=True)
class ScoringConfig:
    metrics_config: Path
    task_paths: list[str]


def experiment_config(config: dict[str, Any]) -> ExperimentConfig:
    values = _section(config, "experiment")
    baseline = values.get("baseline_functionality")
    return ExperimentConfig(
        trials=int(values.get("trials", 1)),
        run_root=Path(values.get("run_root", "runs")),
        baseline_functionality=None if baseline is None else float(baseline),
        functionality_floor_ratio=float(values.get("functionality_floor_ratio", 0.90)),
        daemon_sleep_s=float(values.get("daemon_sleep_s", 30)),
    )


def model_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(_section(config, "model"))


def resources_config(config: dict[str, Any]) -> ResourcesConfig:
    values = dict(_section(config, "resources"))
    return ResourcesConfig(
        max_cpu_threads=int(values.get("max_cpu_threads", 4)),
        task_timeout_s=int(values.get("task_timeout_s", 45)),
        values=values,
    )


def scoring_config(config: dict[str, Any]) -> ScoringConfig:
    values = _section(config, "scoring")
    task_paths = list(values.get("task_paths", []))
    if not task_paths:
        raise SystemExit("No task_paths configured")
    return ScoringConfig(
        metrics_config=Path(values.get("metrics_config", "configs/metrics.yaml")),
        task_paths=[str(path) for path in task_paths],
    )


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    values = config.get(name, {})
    if not isinstance(values, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return values
