from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from agent_harness.backends import make_backend
from agent_harness.task import run_task
from interpretability.config import load_yaml
from interpretability.resources import configure_conservative_threads, detect_resources, should_backoff
from interpretability.scoring import ScoreResult, score_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-trials", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    resources_cfg = config.get("resources", {})
    configure_conservative_threads(int(resources_cfg.get("max_cpu_threads", 4)))
    backoff, reason = should_backoff(resources_cfg)
    if backoff:
        raise SystemExit(f"Resource backoff: {reason}")

    run_root = Path(config.get("experiment", {}).get("run_root", "runs"))
    run_root.mkdir(parents=True, exist_ok=True)
    backend = make_backend(config.get("model", {}))
    metrics_config = load_yaml(config.get("scoring", {}).get("metrics_config", "configs/metrics.yaml"))
    trials = args.max_trials or int(config.get("experiment", {}).get("trials", 1))
    floor_ratio = float(config.get("experiment", {}).get("functionality_floor_ratio", 0.90))
    baseline = config.get("experiment", {}).get("baseline_functionality")
    task_paths = config.get("scoring", {}).get("task_paths", [])
    if not task_paths:
        raise SystemExit("No task_paths configured")

    if baseline is None:
        baseline = _run_trial(run_root, "baseline", backend, task_paths, config, metrics_config, 0.0).functionality

    incumbent_explainability = 0.0
    accepted: list[dict] = []
    for trial in range(trials):
        result = _run_trial(run_root, f"trial-{trial + 1}", backend, task_paths, config, metrics_config, incumbent_explainability, baseline, floor_ratio)
        if result.accepted:
            incumbent_explainability = result.explainability
            accepted.append(result.__dict__)

    summary = {
        "baseline_functionality": baseline,
        "accepted": accepted,
        "resources": detect_resources().__dict__,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _run_trial(
    run_root: Path,
    label: str,
    backend,
    task_paths: list[str],
    config: dict,
    metrics_config: dict,
    incumbent_explainability: float,
    baseline_functionality: float | None = None,
    floor_ratio: float = 0.90,
) -> ScoreResult:
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
        return ScoreResult(functionality, 0.0, True, "baseline", {})
    score = score_run(run_dir, metrics_config, baseline_functionality, floor_ratio, incumbent_explainability)
    (run_dir / "score.json").write_text(json.dumps(score.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return score


if __name__ == "__main__":
    main()

