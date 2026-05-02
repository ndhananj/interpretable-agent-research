from __future__ import annotations

from pathlib import Path

from agent_harness.backends import MockBackend
from interpretability.experiment import prepare_experiment, run_baseline_cycle, run_trial_cycle


def test_shared_experiment_cycles_match_baseline_and_trial_behavior(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
experiment:
  trials: 1
  run_root: {tmp_path / "runs"}
  baseline_functionality: null
  functionality_floor_ratio: 0.90
model:
  backend: mock
resources:
  max_cpu_threads: 2
  max_load_ratio: 99
  min_available_memory_gib: 0
  task_timeout_s: 10
scoring:
  metrics_config: configs/metrics.yaml
  task_paths:
    - tasks/replace_token.yaml
""",
        encoding="utf-8",
    )
    context = prepare_experiment(str(config_path), backend=MockBackend())

    baseline = run_baseline_cycle(context)
    trial = run_trial_cycle(context, baseline.score.functionality, 0.0, 1)

    assert baseline.score.functionality == 1.0
    assert baseline.score.reason == "baseline"
    assert trial.score.functionality == 1.0
    assert "-trial-1-" in trial.run_dir.name
    assert (trial.run_dir / "score.json").exists()
