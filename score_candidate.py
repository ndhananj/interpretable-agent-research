from __future__ import annotations

import argparse
import json

from interpretability.config import load_yaml
from interpretability.scoring import score_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--incumbent-explainability", type=float, default=0.0)
    args = parser.parse_args()

    config = load_yaml(args.config)
    metrics_config = load_yaml(config.get("scoring", {}).get("metrics_config", "configs/metrics.yaml"))
    baseline = config.get("experiment", {}).get("baseline_functionality")
    if baseline is None:
        baseline = 1.0
    floor_ratio = float(config.get("experiment", {}).get("functionality_floor_ratio", 0.90))
    result = score_run(args.run, metrics_config, float(baseline), floor_ratio, args.incumbent_explainability)
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

