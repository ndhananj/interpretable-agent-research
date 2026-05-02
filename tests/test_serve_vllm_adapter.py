from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from interpretability.config import load_yaml
import serve_vllm_adapter
from serve_vllm_adapter import (
    evaluate_dependency_compatibility,
    format_dependency_repair_command,
    _resolve_vllm_executable,
    build_vllm_command,
    format_cuda_preflight_error,
    run_cuda_preflight,
)


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


def test_resolve_vllm_executable_prefers_current_python_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text("")
    vllm = bin_dir / "vllm"
    vllm.write_text("")
    monkeypatch.setattr(serve_vllm_adapter.sys, "executable", str(python))

    assert _resolve_vllm_executable() == str(vllm)


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


def test_dependency_compatibility_accepts_known_good_versions() -> None:
    error = evaluate_dependency_compatibility({"vllm": "0.6.6.post1", "transformers": "4.56.2"})

    assert error is None


def test_dependency_compatibility_rejects_vllm_066_with_transformers_5() -> None:
    error = evaluate_dependency_compatibility({"vllm": "0.6.6.post1", "transformers": "5.7.0"})

    assert error is not None
    assert "vllm==0.6.6.post1 requires transformers<5" in error.reason
    assert "installed transformers is 5.7.0" in error.reason
    assert error.repair_command == "uv pip install 'transformers>=4.56.2,<5'"


def test_dependency_repair_command_quotes_version_range() -> None:
    assert format_dependency_repair_command("transformers>=4.56.2,<5") == (
        "uv pip install 'transformers>=4.56.2,<5'"
    )


def test_cuda_preflight_passes_when_cuda_device_is_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve_vllm_adapter.importlib, "import_module", lambda name: _fake_torch())
    monkeypatch.setattr(serve_vllm_adapter, "_run_nvidia_smi_list", lambda: "GPU 0: Test GPU")
    monkeypatch.setattr(
        serve_vllm_adapter,
        "_get_installed_package_version",
        lambda package: {"vllm": "0.6.6.post1", "transformers": "4.56.2"}[package],
    )
    monkeypatch.setattr(serve_vllm_adapter, "_run_vllm_native_import", lambda: None)

    result = run_cuda_preflight()

    assert result.ok is True
    assert result.python_executable == sys.executable
    assert result.torch_version == "2.4.1+cu121"
    assert result.torch_cuda == "12.1"
    assert result.cuda_available is True
    assert result.device_count == 1
    assert result.vllm_version == "0.6.6.post1"
    assert result.transformers_version == "4.56.2"
    assert result.dependency_error is None
    assert result.vllm_native_import is None


def test_cuda_preflight_fails_fast_on_dependency_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_native_import() -> None:
        raise AssertionError("native import should not run when dependencies are incompatible")

    monkeypatch.setattr(
        serve_vllm_adapter,
        "_get_installed_package_version",
        lambda package: {"vllm": "0.6.6.post1", "transformers": "5.7.0"}[package],
    )
    monkeypatch.setattr(serve_vllm_adapter, "_run_nvidia_smi_list", lambda: "GPU 0: Test GPU")
    monkeypatch.setattr(serve_vllm_adapter, "_run_vllm_native_import", _raise_native_import)

    result = run_cuda_preflight()
    message = format_cuda_preflight_error(result)

    assert result.ok is False
    assert result.vllm_version == "0.6.6.post1"
    assert result.transformers_version == "5.7.0"
    assert result.dependency_error is not None
    assert "Python package dependency mismatch" in message
    assert "vLLM: 0.6.6.post1" in message
    assert "transformers: 5.7.0" in message
    assert "uv pip install 'transformers>=4.56.2,<5'" in message
    assert "python serve_vllm_adapter.py --config configs/vllm.yaml" in message
    assert "pip uninstall -y vllm torch" not in message


def test_cuda_preflight_fails_when_vllm_native_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve_vllm_adapter.importlib, "import_module", lambda name: _fake_torch())
    monkeypatch.setattr(serve_vllm_adapter, "_run_nvidia_smi_list", lambda: "GPU 0: Test GPU")
    monkeypatch.setattr(serve_vllm_adapter, "_get_installed_package_version", lambda package: "0.20.0")
    monkeypatch.setattr(
        serve_vllm_adapter,
        "_run_vllm_native_import",
        lambda: "import vllm._C failed: ImportError: libcudart.so.13: cannot open shared object file",
    )

    result = run_cuda_preflight()
    message = format_cuda_preflight_error(result)

    assert result.ok is False
    assert "python: " in message
    assert "torch: 2.4.1+cu121" in message
    assert "torch.version.cuda: 12.1" in message
    assert "vLLM: 0.20.0" in message
    assert "nvidia-smi -L: GPU 0: Test GPU" in message
    assert "vLLM native import: import vllm._C failed: ImportError: libcudart.so.13" in message
    assert "reason: vLLM native extension import failed" in message
    assert "Fix the NVIDIA driver first" in message
    assert "uv pip install vllm==0.6.6.post1 --torch-backend=cu121" in message


def test_cuda_preflight_fails_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        serve_vllm_adapter.importlib,
        "import_module",
        lambda name: _fake_torch(available=False, device_count=0),
    )
    monkeypatch.setattr(serve_vllm_adapter, "_run_nvidia_smi_list", lambda: "failed: driver API mismatch")
    monkeypatch.setattr(serve_vllm_adapter, "_get_installed_package_version", lambda package: "0.20.0")
    monkeypatch.setattr(
        serve_vllm_adapter,
        "_run_vllm_native_import",
        lambda: "import vllm._C failed: ImportError: libcudart.so.13: cannot open shared object file",
    )

    result = run_cuda_preflight()
    message = format_cuda_preflight_error(result)

    assert result.ok is False
    assert "torch reports CUDA is unavailable" in message
    assert "torch: 2.4.1+cu121" in message
    assert "torch.version.cuda: 12.1" in message
    assert "nvidia-smi -L: failed: driver API mismatch" in message
    assert "vLLM: 0.20.0" in message
    assert "libcudart.so.13" in message
    assert "Recommended repair order" in message
    assert "Fix or reload the NVIDIA driver until `nvidia-smi` works" in message


def test_cuda_preflight_fails_when_torch_import_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_import_error(name: str) -> None:
        raise ImportError("missing torch")

    monkeypatch.setattr(serve_vllm_adapter.importlib, "import_module", _raise_import_error)
    monkeypatch.setattr(serve_vllm_adapter, "_run_nvidia_smi_list", lambda: "not found")
    monkeypatch.setattr(serve_vllm_adapter, "_get_installed_package_version", lambda package: "not installed")
    monkeypatch.setattr(
        serve_vllm_adapter,
        "_run_vllm_native_import",
        lambda: "import vllm._C failed: ModuleNotFoundError: No module named 'vllm'",
    )

    result = run_cuda_preflight()
    message = format_cuda_preflight_error(result)

    assert result.ok is False
    assert "torch import failed: missing torch" in message
    assert "torch: unknown" in message
    assert "vLLM: not installed" in message
    assert "No module named 'vllm'" in message


def test_cuda_preflight_fails_when_no_cuda_devices_are_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        serve_vllm_adapter.importlib,
        "import_module",
        lambda name: _fake_torch(available=True, device_count=0),
    )
    monkeypatch.setattr(serve_vllm_adapter, "_run_nvidia_smi_list", lambda: "GPU 0: Test GPU")
    monkeypatch.setattr(serve_vllm_adapter, "_get_installed_package_version", lambda package: "0.6.6.post1")
    monkeypatch.setattr(serve_vllm_adapter, "_run_vllm_native_import", lambda: None)

    result = run_cuda_preflight()
    message = format_cuda_preflight_error(result)

    assert result.ok is False
    assert "torch reports zero visible CUDA devices" in message
    assert "torch.cuda.device_count(): 0" in message


def test_vllm_native_import_summarizes_raw_traceback() -> None:
    output = """
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ImportError: libcudart.so.13: cannot open shared object file: No such file or directory
"""

    message = serve_vllm_adapter._summarize_import_failure(output, 1)

    assert message == (
        "import vllm._C failed: ImportError: "
        "libcudart.so.13: cannot open shared object file: No such file or directory"
    )


def test_main_dry_run_skips_runtime_preflight(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def _raise_preflight() -> None:
        raise AssertionError("preflight should not run during dry-run")

    monkeypatch.setattr(sys, "argv", ["serve_vllm_adapter.py", "--config", "configs/vllm.yaml", "--dry-run"])
    monkeypatch.setattr(serve_vllm_adapter, "load_yaml", lambda path: {"model": {"backend": "vllm_openai"}})
    monkeypatch.setattr(
        serve_vllm_adapter,
        "build_vllm_command",
        lambda *args, **kwargs: ["vllm", "serve", "model", "--port", "8000"],
    )
    monkeypatch.setattr(serve_vllm_adapter, "run_cuda_preflight", _raise_preflight)

    serve_vllm_adapter.main()

    assert capsys.readouterr().out.strip() == "vllm serve model --port 8000"


def test_main_skip_preflight_launches_raw_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _raise_preflight() -> None:
        raise AssertionError("preflight should not run with --skip-preflight")

    def _fake_call(cmd: list[str]) -> int:
        calls.append(cmd)
        return 7

    monkeypatch.setattr(sys, "argv", ["serve_vllm_adapter.py", "--config", "configs/vllm.yaml", "--skip-preflight"])
    monkeypatch.setattr(serve_vllm_adapter, "load_yaml", lambda path: {"model": {"backend": "vllm_openai"}})
    monkeypatch.setattr(
        serve_vllm_adapter,
        "build_vllm_command",
        lambda *args, **kwargs: ["vllm", "serve", "model", "--port", "8000"],
    )
    monkeypatch.setattr(serve_vllm_adapter, "run_cuda_preflight", _raise_preflight)
    monkeypatch.setattr(serve_vllm_adapter, "_resolve_vllm_executable", lambda: "/venv/bin/vllm")
    monkeypatch.setattr(serve_vllm_adapter.subprocess, "call", _fake_call)

    with pytest.raises(SystemExit) as exc_info:
        serve_vllm_adapter.main()

    assert exc_info.value.code == 7
    assert calls == [["/venv/bin/vllm", "serve", "model", "--port", "8000"]]
