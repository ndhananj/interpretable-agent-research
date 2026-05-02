from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_harness.backends import make_backend
from agent_harness.task import run_task
from interpretability.artifacts import write_json
from interpretability.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", default="runs/manual-task")
    args = parser.parse_args()

    config = load_yaml(args.config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    backend = make_backend(config.get("model", {}))
    timeout_s = int(config.get("resources", {}).get("task_timeout_s", 45))
    result = run_task(args.task, backend, run_dir, timeout_s)
    metrics = {"functionality": result.functionality}
    write_json(run_dir / "metrics.json", metrics, sort_keys=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
