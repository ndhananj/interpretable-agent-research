from __future__ import annotations

import argparse
import json

from interpretability.experiment import (
    ExperimentContext,
    ExperimentResult,
    legacy_summary,
    prepare_experiment,
    run_baseline_cycle as _run_baseline_cycle,
    run_experiment,
    run_trial,
    run_trial_cycle,
    write_legacy_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-trials", type=int, default=None)
    args = parser.parse_args()

    summary = run_experiment(args.config, args.max_trials)
    print(json.dumps(legacy_summary(summary), indent=2, sort_keys=True))


def run_one_trial_cycle(*args, **kwargs):
    """Compatibility wrapper for older imports."""
    if args and isinstance(args[0], ExperimentContext):
        return run_trial_cycle(*args, **kwargs)
    run_root, backend, task_paths, config, metrics_config, baseline, incumbent, trial_number = args[:8]
    floor_ratio = args[8] if len(args) > 8 else kwargs.get("floor_ratio", 0.90)
    context = ExperimentContext(
        config=config,
        run_root=run_root,
        backend=backend,
        metrics_config=metrics_config,
        task_paths=task_paths,
        floor_ratio=floor_ratio,
        baseline_functionality=baseline,
        task_timeout_s=int(config.get("resources", {}).get("task_timeout_s", 45)),
        trials=int(config.get("experiment", {}).get("trials", 1)),
    )
    record = run_trial_cycle(context, baseline, incumbent, trial_number)
    return record.score, record.run_dir


def run_baseline_cycle(*args, **kwargs):
    if args and isinstance(args[0], ExperimentContext):
        return _run_baseline_cycle(*args, **kwargs)
    run_root, backend, task_paths, config, metrics_config = args[:5]
    context = ExperimentContext(
        config=config,
        run_root=run_root,
        backend=backend,
        metrics_config=metrics_config,
        task_paths=task_paths,
        floor_ratio=float(config.get("experiment", {}).get("functionality_floor_ratio", 0.90)),
        baseline_functionality=None,
        task_timeout_s=int(config.get("resources", {}).get("task_timeout_s", 45)),
        trials=int(config.get("experiment", {}).get("trials", 1)),
    )
    record = _run_baseline_cycle(context)
    return record.score, record.run_dir


if __name__ == "__main__":
    main()
