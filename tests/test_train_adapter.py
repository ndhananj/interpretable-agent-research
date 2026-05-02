from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_train_adapter_dry_run_validates_config_and_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "examples.jsonl"
    dataset.write_text('{"prompt":"p","response":"r"}\n', encoding="utf-8")
    config = tmp_path / "adapter.yaml"
    config.write_text(
        "\n".join(
            [
                "model_name: Qwen/Qwen2.5-Coder-1.5B-Instruct",
                f"output_dir: {tmp_path / 'adapter'}",
                f"dataset_path: {dataset}",
                "lora:",
                "  r: 8",
                "  alpha: 16",
                "  dropout: 0.05",
                "  target_modules:",
                "    - q_proj",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "train_adapter.py", "--config", str(config), "--dry-run"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "with 1 examples" in proc.stdout


def test_train_adapter_dry_run_rejects_missing_dataset(tmp_path: Path) -> None:
    config = tmp_path / "adapter.yaml"
    config.write_text(
        "\n".join(
            [
                "model_name: Qwen/Qwen2.5-Coder-1.5B-Instruct",
                f"output_dir: {tmp_path / 'adapter'}",
                f"dataset_path: {tmp_path / 'missing.jsonl'}",
                "lora:",
                "  target_modules:",
                "    - q_proj",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "train_adapter.py", "--config", str(config), "--dry-run"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "Dataset not found" in proc.stderr
