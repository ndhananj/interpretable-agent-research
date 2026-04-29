from __future__ import annotations

import argparse
from pathlib import Path

from interpretability.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adapter.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.dry_run:
        print(f"Adapter dry run for {config['model_name']} -> {config['output_dir']}")
        return
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Adapter training requires optional dependencies from requirements.txt. "
            "Run: pip install -r requirements.txt"
        ) from exc
    dataset_path = Path(config["dataset_path"])
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}. Add JSONL training examples before training.")
    raise SystemExit("Training implementation placeholder: wire SFTTrainer/LoRA after dataset format is finalized.")


if __name__ == "__main__":
    main()

