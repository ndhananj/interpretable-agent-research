from __future__ import annotations

import argparse
from pathlib import Path

from interpretability.artifacts import read_json
from interpretability.config import load_yaml, model_config
from interpretability.mechanistic import write_mechanistic_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate a mechanistic interpretability report for a run")
    parser.add_argument("--config", default="configs/mechanistic.yaml")
    parser.add_argument("--run", required=True, help="Run directory containing task artifacts")
    parser.add_argument("--model-config", default=None, help="Optional experiment/vLLM config for model metadata")
    args = parser.parse_args()

    model = {}
    if args.model_config:
        model = model_config(load_yaml(args.model_config))
    else:
        model = read_json(Path(args.run) / "config.json", {}).get("model", {})

    report = write_mechanistic_report(Path(args.run), args.config, model)
    print(f"Wrote {Path(args.run) / 'mechanistic.json'} with {len(report.evidence)} evidence records")


if __name__ == "__main__":
    main()

