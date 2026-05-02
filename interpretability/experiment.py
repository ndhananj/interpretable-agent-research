from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_harness.backends import ModelBackend, make_backend
from agent_harness.task import run_task
from interpretability.config import (
    experiment_config,
    load_yaml,
    model_config,
    resources_config,
    scoring_config,
)
from interpretability.resources import configure_conservative_threads, detect_resources, should_backoff
from interpretability.scoring import ScoreResult, score_run


@dataclass(frozen=True)
class ExperimentContext:
    config: dict[str, Any]
    run_root: Path
    backend: ModelBackend
    metrics_config: dict[str, Any]
    task_paths: list[str]
    floor_ratio: float
    baseline_functionality: float | None
    task_timeout_s: int
    trials: int


@dataclass(frozen=True)
class TrialRecord:
    label: str
    run_dir: Path
    score: ScoreResult


@dataclass(frozen=True)
class ExperimentResult:
    baseline_functionality: float
    accepted: list[dict[str, Any]]
    resources: dict[str, Any]
    trials: list[dict[str, Any]]


def prepare_experiment(config_path: str, backend: ModelBackend | None = None) -> ExperimentContext:
    config = load_yaml(config_path)
    exp = experiment_config(config)
    resources = resources_config(config)
    scoring = scoring_config(config)

    configure_conservative_threads(resources.max_cpu_threads)
    backoff, reason = should_backoff(resources.values)
    if backoff:
        raise SystemExit(f"Resource backoff: {reason}")

    exp.run_root.mkdir(parents=True, exist_ok=True)
    return ExperimentContext(
        config=config,
        run_root=exp.run_root,
        backend=backend or make_backend(model_config(config)),
        metrics_config=load_yaml(scoring.metrics_config),
        task_paths=scoring.task_paths,
        floor_ratio=exp.functionality_floor_ratio,
        baseline_functionality=exp.baseline_functionality,
        task_timeout_s=resources.task_timeout_s,
        trials=exp.trials,
    )


def run_experiment(config_path: str = "configs/default.yaml", max_trials: int | None = None) -> ExperimentResult:
    context = prepare_experiment(config_path)
    records, baseline, incumbent, accepted = ensure_baseline(
        context,
        context.baseline_functionality,
        incumbent_explainability=0.0,
        accepted=[],
    )

    for trial in range(max_trials or context.trials):
        record = run_trial_cycle(context, baseline, incumbent, trial + 1)
        records.append(record)
        if record.score.accepted:
            incumbent = record.score.explainability
            accepted.append(record.score.__dict__)

    summary = ExperimentResult(
        baseline_functionality=baseline,
        accepted=accepted,
        resources=detect_resources().__dict__,
        trials=[trial_record_payload(record) for record in records],
    )
    write_legacy_summary(context.run_root, summary)
    return summary


def ensure_baseline(
    context: ExperimentContext,
    baseline_functionality: float | None,
    incumbent_explainability: float,
    accepted: list[dict[str, Any]],
) -> tuple[list[TrialRecord], float, float, list[dict[str, Any]]]:
    if baseline_functionality is not None:
        return [], baseline_functionality, incumbent_explainability, accepted

    record = run_baseline_cycle(context)
    return [record], record.score.functionality, incumbent_explainability, accepted


def run_baseline_cycle(context: ExperimentContext) -> TrialRecord:
    score, run_dir = run_trial(context, "baseline", incumbent_explainability=0.0)
    return TrialRecord("baseline", run_dir, score)


def run_trial_cycle(
    context: ExperimentContext,
    baseline_functionality: float,
    incumbent_explainability: float,
    trial_number: int,
) -> TrialRecord:
    label = f"trial-{trial_number}"
    score, run_dir = run_trial(
        context,
        label,
        incumbent_explainability=incumbent_explainability,
        baseline_functionality=baseline_functionality,
    )
    return TrialRecord(label, run_dir, score)


def run_trial(
    context: ExperimentContext,
    label: str,
    incumbent_explainability: float,
    baseline_functionality: float | None = None,
) -> tuple[ScoreResult, Path]:
    run_dir = context.run_root / f"{time.strftime('%Y%m%dT%H%M%S')}-{label}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    scores = [
        run_task(task_path, context.backend, run_dir, context.task_timeout_s).functionality
        for task_path in context.task_paths
    ]
    functionality = sum(scores) / max(len(scores), 1)
    (run_dir / "metrics.json").write_text(json.dumps({"functionality": functionality}, indent=2) + "\n", encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(context.config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "mechanistic.json").write_text(json.dumps({"sparsity": 0.30, "stability": 0.30}, indent=2) + "\n", encoding="utf-8")
    if baseline_functionality is None:
        return ScoreResult(functionality, 0.0, True, "baseline", {}), run_dir
    score = score_run(
        run_dir,
        context.metrics_config,
        baseline_functionality,
        context.floor_ratio,
        incumbent_explainability,
    )
    (run_dir / "score.json").write_text(json.dumps(score.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return score, run_dir


def trial_record_payload(record: TrialRecord) -> dict[str, Any]:
    return {
        "label": record.label,
        "run_dir": str(record.run_dir),
        "functionality": record.score.functionality,
        "explainability": record.score.explainability,
        "accepted": record.score.accepted,
        "reason": record.score.reason,
    }


def legacy_summary(result: ExperimentResult) -> dict[str, Any]:
    return {
        "baseline_functionality": result.baseline_functionality,
        "accepted": result.accepted,
        "resources": result.resources,
    }


def write_legacy_summary(run_root: Path, result: ExperimentResult) -> None:
    (run_root / "summary.json").write_text(
        json.dumps(legacy_summary(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_summary(run_root: Path, baseline: float, accepted: list[dict[str, Any]], resources: dict[str, Any]) -> None:
    write_legacy_summary(
        run_root,
        ExperimentResult(baseline_functionality=baseline, accepted=accepted, resources=resources, trials=[]),
    )
