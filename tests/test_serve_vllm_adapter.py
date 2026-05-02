from __future__ import annotations

from pathlib import Path

import pytest

from interpretability.config import load_yaml
from serve_vllm_adapter import build_vllm_command


def test_build_vllm_command_from_config_dry_run() -> None:
    config = load_yaml("configs/vllm.yaml")

    cmd = build_vllm_command(config, config_path="configs/vllm.yaml", require_adapter_path_exists=False)

    assert cmd == [
        "vllm",
        "serve",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "--enable-lora",
        "--lora-modules",
        f"interpretable-agent-lora={Path.cwd() / 'adapters/latest'}",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]


def test_build_vllm_command_accepts_host_port_overrides() -> None:
    config = load_yaml("configs/vllm.yaml")

    cmd = build_vllm_command(
        config,
        config_path="configs/vllm.yaml",
        host="0.0.0.0",
        port=8010,
        require_adapter_path_exists=False,
    )

    assert cmd[-4:] == ["--host", "0.0.0.0", "--port", "8010"]


def test_build_vllm_command_rejects_missing_adapter_path() -> None:
    config = load_yaml("configs/vllm.yaml")
    config["model"]["adapter_path"] = ""

    with pytest.raises(SystemExit, match="model.adapter_path"):
        build_vllm_command(config, config_path="configs/vllm.yaml", require_adapter_path_exists=False)


def test_build_vllm_command_rejects_nonexistent_adapter_path_before_launch(tmp_path: Path) -> None:
    config = load_yaml("configs/vllm.yaml")
    config["model"]["adapter_path"] = str(tmp_path / "missing-adapter")

    with pytest.raises(SystemExit, match="does not exist"):
        build_vllm_command(config, config_path="configs/vllm.yaml", require_adapter_path_exists=True)
