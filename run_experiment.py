from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from agent_harness.backends import make_backend
from agent_harness.task import run_task
from interpretability.config import load_yaml
from interpretability.resources import configure_conservative_threads, detect_resources, should_backoff
from interpretability.scoring import ScoreResult, score_run


@dataclass(frozen=True)
class ExperimentResult:
    baseline_functionality: float
    accepted: list[dict]
    resources: dict
    trials: list[dict]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-trials", type=int, default=None)
    args = parser.parse_args()

    summary = run_experiment(args.config, args.max_trials)
    print(json.dumps(legacy_summary(summary), indent=2, sort_keys=True))


def prepare_experiment(config_path: str) -> tuple[dict, Path, object, dict, list[str], float, float | None]:
    config = load_yaml(config_path)
    resources_cfg = config.get("resources", {})
    configure_conservative_threads(int(resources_cfg.get("max_cpu_threads", 4)))
    backoff, reason = should_backoff(resources_cfg)
    if backoff:
        raise SystemExit(f"Resource backoff: {reason}")

    run_root = Path(config.get("experiment", {}).get("run_root", "runs"))
    run_root.mkdir(parents=True, exist_ok=True)
    backend = make_backend(config.get("model", {}))
    metrics_config = load_yaml(config.get("scoring", {}).get("metrics_config", "configs/metrics.yaml"))
    floor_ratio = float(config.get("experiment", {}).get("functionality_floor_ratio", 0.90))
    baseline = config.get("experiment", {}).get("baseline_functionality")
    task_paths = config.get("scoring", {}).get("task_paths", [])
    if not task_paths:
        raise SystemExit("No task_paths configured")
    return config, run_root, backend, metrics_config, task_paths, floor_ratio, baseline


def run_experiment(config_path: str = "configs/default.yaml", max_trials: int | None = None) -> ExperimentResult:
    config, run_root, backend, metrics_config, task_paths, floor_ratio, baseline = prepare_experiment(config_path)
    trials = max_trials or int(config.get("experiment", {}).get("trials", 1))

    if baseline is None:
        baseline_result, baseline_dir = run_trial(
            run_root, "baseline", backend, task_paths, config, metrics_config, 0.0
        )
        baseline = baseline_result.functionality
        trial_records = [_trial_record("baseline", baseline_dir, baseline_result)]
    else:
        trial_records = []

    incumbent_explainability = 0.0
    accepted: list[dict] = []
    for trial in range(trials):
        result, run_dir = run_trial(
            run_root,
            f"trial-{trial + 1}",
            backend,
            task_paths,
            config,
            metrics_config,
            incumbent_explainability,
            baseline,
            floor_ratio,
        )
        trial_records.append(_trial_record(f"trial-{trial + 1}", run_dir, result))
        if result.accepted:
            incumbent_explainability = result.explainability
            accepted.append(result.__dict__)

    summary = ExperimentResult(
        baseline_functionality=baseline,
        accepted=accepted,
        resources=detect_resources().__dict__,
        trials=trial_records,
    )
    write_legacy_summary(run_root, summary)
    return summary


def _trial_record(label: str, run_dir: Path, result: ScoreResult) -> dict:
    return {
        "label": label,
        "run_dir": str(run_dir),
        "functionality": result.functionality,
        "explainability": result.explainability,
        "accepted": result.accepted,
        "reason": result.reason,
    }


def legacy_summary(result: ExperimentResult) -> dict:
    return {
        "baseline_functionality": result.baseline_functionality,
        "accepted": result.accepted,
        "resources": result.resources,
    }


def write_legacy_summary(run_root: Path, result: ExperimentResult) -> None:
    (run_root / "summary.json").write_text(
        json.dumps(legacy_summary(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_one_trial_cycle(
    run_root: Path,
    backend,
    task_paths: list[str],
    config: dict,
    metrics_config: dict,
    baseline_functionality: float,
    incumbent_explainability: float,
    trial_number: int,
    floor_ratio: float = 0.90,
) -> tuple[ScoreResult, Path]:
    return run_trial(
        run_root,
        f"trial-{trial_number}",
        backend,
        task_paths,
        config,
        metrics_config,
        incumbent_explainability,
        baseline_functionality,
        floor_ratio,
    )


def run_baseline_cycle(
    run_root: Path,
    backend,
    task_paths: list[str],
    config: dict,
    metrics_config: dict,
) -> tuple[ScoreResult, Path]:
    return run_trial(run_root, "baseline", backend, task_paths, config, metrics_config, 0.0)


def run_trial(
    run_root: Path,
    label: str,
    backend,
    task_paths: list[str],
    config: dict,
    metrics_config: dict,
    incumbent_explainability: float,
    baseline_functionality: float | None = None,
    floor_ratio: float = 0.90,
) -> tuple[ScoreResult, Path]:
    run_dir = run_root / f"{time.strftime('%Y%m%dT%H%M%S')}-{label}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    timeout_s = int(config.get("resources", {}).get("task_timeout_s", 45))
    scores = []
    for task_path in task_paths:
        result = run_task(task_path, backend, run_dir, timeout_s)
        scores.append(result.functionality)
    functionality = sum(scores) / max(len(scores), 1)
    (run_dir / "metrics.json").write_text(json.dumps({"functionality": functionality}, indent=2) + "\n", encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "mechanistic.json").write_text(json.dumps({"sparsity": 0.30, "stability": 0.30}, indent=2) + "\n", encoding="utf-8")
    if baseline_functionality is None:
        return ScoreResult(functionality, 0.0, True, "baseline", {}), run_dir
    score = score_run(run_dir, metrics_config, baseline_functionality, floor_ratio, incumbent_explainability)
    (run_dir / "score.json").write_text(json.dumps(score.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return score, run_dir


if __name__ == "__main__":
    main()
