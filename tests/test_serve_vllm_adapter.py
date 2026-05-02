from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from interpretability.config import load_yaml
import serve_vllm_adapter
from serve_vllm_adapter import build_vllm_command, format_cuda_preflight_error, run_cuda_preflight


class _FakeCuda:
    def __init__(self, *, available: bool, device_count: int) -> None:
        self._available = available
        self._device_count = device_count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._device_count


def _fake_torch(*, cuda: str | None = "12.1", available: bool = True, device_count: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        __version__="2.4.1+cu121",
        version=SimpleNamespace(cuda=cuda),
        cuda=_FakeCuda(available=available, device_count=device_count),
    )


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


def test_cuda_preflight_passes_when_cuda_device_is_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve_vllm_adapter.importlib, "import_module", lambda name: _fake_torch())
    monkeypatch.setattr(serve_vllm_adapter, "_run_nvidia_smi_list", lambda: "GPU 0: Test GPU")

    result = run_cuda_preflight()

    assert result.ok is True
    assert result.torch_version == "2.4.1+cu121"
    assert result.torch_cuda == "12.1"
    assert result.cuda_available is True
    assert result.device_count == 1


def test_cuda_preflight_fails_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        serve_vllm_adapter.importlib,
        "import_module",
        lambda name: _fake_torch(available=False, device_count=0),
    )
    monkeypatch.setattr(serve_vllm_adapter, "_run_nvidia_smi_list", lambda: "failed: driver API mismatch")

    result = run_cuda_preflight()
    message = format_cuda_preflight_error(result)

    assert result.ok is False
    assert "torch reports CUDA is unavailable" in message
    assert "torch: 2.4.1+cu121" in message
    assert "torch.version.cuda: 12.1" in message
    assert "nvidia-smi -L: failed: driver API mismatch" in message


def test_cuda_preflight_fails_when_torch_import_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_import_error(name: str) -> None:
        raise ImportError("missing torch")

    monkeypatch.setattr(serve_vllm_adapter.importlib, "import_module", _raise_import_error)

    result = run_cuda_preflight()
    message = format_cuda_preflight_error(result)

    assert result.ok is False
    assert "torch import failed: missing torch" in message
    assert "torch: unknown" in message


def test_cuda_preflight_fails_when_no_cuda_devices_are_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        serve_vllm_adapter.importlib,
        "import_module",
        lambda name: _fake_torch(available=True, device_count=0),
    )
    monkeypatch.setattr(serve_vllm_adapter, "_run_nvidia_smi_list", lambda: "GPU 0: Test GPU")

    result = run_cuda_preflight()
    message = format_cuda_preflight_error(result)

    assert result.ok is False
    assert "torch reports zero visible CUDA devices" in message
    assert "torch.cuda.device_count(): 0" in message
