from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path
from types import SimpleNamespace

from train_adapter import _select_training_runtime


def _fake_torch(cuda_available: bool) -> SimpleNamespace:
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda_available))


def test_select_training_runtime_uses_cpu_when_cuda_unavailable() -> None:
    runtime = _select_training_runtime(_fake_torch(False))

    assert runtime == {"model_kwargs": {}, "bf16": False, "fp16": False, "use_cpu": True}


def test_select_training_runtime_uses_fp16_gpu_when_cuda_available() -> None:
    runtime = _select_training_runtime(_fake_torch(True))

    assert runtime == {
        "model_kwargs": {"device_map": "auto"},
        "bf16": False,
        "fp16": True,
        "use_cpu": False,
    }


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
    metadata = json.loads((tmp_path / "adapter" / "training_preflight.json").read_text(encoding="utf-8"))
    assert metadata["dataset_size"] == 1
    assert "tiny dataset: 1 example(s)" in metadata["warnings"]
    assert "prompt-shape mismatch: expected real harness workspace snapshot prompts" in metadata["warnings"]


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


def test_train_adapter_preflight_detects_harness_prompt_shape(tmp_path: Path) -> None:
    dataset = tmp_path / "examples.jsonl"
    dataset.write_text(
        '{"prompt":"Task instruction:\\nDo it.\\n\\nWorking directory: /tmp/work\\nWorkspace files:\\n--- input.txt ---\\nTODO\\n\\nProduce the next complete action as JSON","response":"{}"}\n',
        encoding="utf-8",
    )
    guardrail = tmp_path / "guardrail.jsonl"
    guardrail.write_text(dataset.read_text(encoding="utf-8"), encoding="utf-8")
    config = tmp_path / "adapter.yaml"
    config.write_text(
        "\n".join(
            [
                "model_name: base",
                f"output_dir: {tmp_path / 'adapter'}",
                f"dataset_path: {dataset}",
                f"guardrail_dataset_path: {guardrail}",
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

    assert proc.returncode == 0, proc.stderr
    metadata = json.loads((tmp_path / "adapter" / "training_preflight.json").read_text(encoding="utf-8"))
    assert "prompt-shape mismatch: expected real harness workspace snapshot prompts" not in metadata["warnings"]
    assert metadata["guardrail_dataset_exists"] is True
