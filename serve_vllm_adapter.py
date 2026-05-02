from __future__ import annotations

import argparse
import importlib
import shlex
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from interpretability.config import load_yaml


@dataclass(frozen=True)
class CudaPreflightResult:
    ok: bool
    torch_version: str = "unknown"
    torch_cuda: str = "unknown"
    cuda_available: bool = False
    device_count: int | None = None
    nvidia_smi: str | None = None
    error: str | None = None


def _host_port_from_base_url(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SystemExit("model.base_url must be an http(s) URL with a host")
    if parsed.port is None:
        return parsed.hostname, 443 if parsed.scheme == "https" else 80
    return parsed.hostname, parsed.port


def build_vllm_command(
    config: dict[str, Any],
    *,
    config_path: str | Path = "configs/vllm.yaml",
    host: str | None = None,
    port: int | None = None,
    require_adapter_path_exists: bool = True,
) -> list[str]:
    model = config.get("model")
    if not isinstance(model, dict):
        raise SystemExit("Config must include a model mapping")
    if model.get("backend") != "vllm_openai":
        raise SystemExit("model.backend must be vllm_openai")

    base_model = model.get("name")
    adapter_name = model.get("adapter_name")
    adapter_path = model.get("adapter_path")
    base_url = model.get("base_url")
    if not isinstance(base_model, str) or not base_model.strip():
        raise SystemExit("model.name must be a non-empty base model name")
    if not isinstance(adapter_name, str) or not adapter_name.strip():
        raise SystemExit("model.adapter_name must be a non-empty LoRA module name")
    if not isinstance(adapter_path, str) or not adapter_path.strip():
        raise SystemExit("model.adapter_path must be a non-empty path to a LoRA adapter")
    if not isinstance(base_url, str) or not base_url.strip():
        raise SystemExit("model.base_url must be a non-empty URL")

    default_host, default_port = _host_port_from_base_url(base_url)
    serve_host = host or default_host
    serve_port = int(port if port is not None else default_port)
    if serve_port <= 0 or serve_port > 65535:
        raise SystemExit("vLLM port must be between 1 and 65535")

    resolved_adapter_path = _resolve_adapter_path(adapter_path)
    if require_adapter_path_exists and not resolved_adapter_path.exists():
        raise SystemExit(f"model.adapter_path does not exist: {resolved_adapter_path}")

    return [
        "vllm",
        "serve",
        base_model,
        "--enable-lora",
        "--lora-modules",
        f"{adapter_name}={resolved_adapter_path}",
        "--host",
        serve_host,
        "--port",
        str(serve_port),
    ]


def _resolve_adapter_path(adapter_path: str) -> Path:
    path = Path(adapter_path).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def run_cuda_preflight() -> CudaPreflightResult:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return CudaPreflightResult(ok=False, error=f"torch import failed: {exc}")

    torch_version = str(getattr(torch, "__version__", "unknown"))
    version = getattr(torch, "version", None)
    torch_cuda_value = getattr(version, "cuda", None) if version is not None else None
    torch_cuda = str(torch_cuda_value or "not available")
    if not torch_cuda_value:
        return CudaPreflightResult(
            ok=False,
            torch_version=torch_version,
            torch_cuda=torch_cuda,
            error="installed torch does not include CUDA support",
        )

    cuda_warning = None
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            cuda_available = bool(torch.cuda.is_available())
        if caught_warnings:
            cuda_warning = str(caught_warnings[-1].message)
    except Exception as exc:
        return CudaPreflightResult(
            ok=False,
            torch_version=torch_version,
            torch_cuda=torch_cuda,
            error=f"torch.cuda.is_available() failed: {exc}",
        )

    device_count: int | None = None
    if cuda_available:
        try:
            device_count = int(torch.cuda.device_count())
        except Exception as exc:
            return CudaPreflightResult(
                ok=False,
                torch_version=torch_version,
                torch_cuda=torch_cuda,
                cuda_available=cuda_available,
                error=f"torch.cuda.device_count() failed: {exc}",
            )

    nvidia_smi = _run_nvidia_smi_list()
    if not cuda_available:
        error = "torch reports CUDA is unavailable"
        if cuda_warning:
            error = f"{error}: {cuda_warning}"
        return CudaPreflightResult(
            ok=False,
            torch_version=torch_version,
            torch_cuda=torch_cuda,
            cuda_available=False,
            device_count=device_count,
            nvidia_smi=nvidia_smi,
            error=error,
        )
    if device_count is None or device_count < 1:
        return CudaPreflightResult(
            ok=False,
            torch_version=torch_version,
            torch_cuda=torch_cuda,
            cuda_available=True,
            device_count=device_count,
            nvidia_smi=nvidia_smi,
            error="torch reports zero visible CUDA devices",
        )
    if nvidia_smi is not None and nvidia_smi.startswith("failed:"):
        return CudaPreflightResult(
            ok=False,
            torch_version=torch_version,
            torch_cuda=torch_cuda,
            cuda_available=True,
            device_count=device_count,
            nvidia_smi=nvidia_smi,
            error="nvidia-smi -L failed",
        )
    return CudaPreflightResult(
        ok=True,
        torch_version=torch_version,
        torch_cuda=torch_cuda,
        cuda_available=True,
        device_count=device_count,
        nvidia_smi=nvidia_smi,
    )


def _run_nvidia_smi_list() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return "not found"
    except Exception as exc:
        return f"failed: {exc}"
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        return f"failed: {output or f'exit code {completed.returncode}'}"
    return output or "ok"


def _resolve_vllm_executable() -> str:
    executable = Path(sys.executable)
    candidates = [executable.parent / "vllm", executable.resolve().parent / "vllm"]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "vllm"


def format_cuda_preflight_error(result: CudaPreflightResult) -> str:
    lines = [
        "vLLM CUDA preflight failed; not starting vllm serve.",
        f"torch: {result.torch_version}",
        f"torch.version.cuda: {result.torch_cuda}",
        f"torch.cuda.is_available(): {result.cuda_available}",
    ]
    if result.device_count is not None:
        lines.append(f"torch.cuda.device_count(): {result.device_count}")
    if result.nvidia_smi is not None:
        lines.append(f"nvidia-smi -L: {result.nvidia_smi}")
    if result.error:
        lines.append(f"reason: {result.error}")
    lines.extend(
        [
            "",
            "Default keep-driver reinstall path:",
            "  python3 -m venv .venv-vllm",
            "  . .venv-vllm/bin/activate",
            "  pip install -r requirements.txt",
            "  uv pip install vllm --torch-backend=auto",
            "",
            "If reusing an existing vLLM environment, remove incompatible CUDA wheels first:",
            "  pip uninstall -y vllm torch torchvision torchaudio 'nvidia-*'",
            "  uv pip install vllm --torch-backend=auto",
            "",
            "Alternate driver-update path:",
            "  Upgrade the NVIDIA driver to support the installed torch CUDA runtime, then reboot or reload the driver.",
            "",
            "Do not use plain `pip install -r requirements-vllm.txt` as the primary GPU fix on this machine.",
            "Use --skip-preflight only when debugging raw vLLM startup behavior.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a configured LoRA adapter through local vLLM")
    parser.add_argument("--config", default="configs/vllm.yaml")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip CUDA checks before launching vLLM; intended for advanced debugging only.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    cmd = build_vllm_command(
        config,
        config_path=args.config,
        host=args.host,
        port=args.port,
        require_adapter_path_exists=not args.dry_run,
    )
    if args.dry_run:
        print(shlex.join(cmd))
        return
    if not args.skip_preflight:
        preflight = run_cuda_preflight()
        if not preflight.ok:
            print(format_cuda_preflight_error(preflight), file=sys.stderr)
            raise SystemExit(1)
    cmd = [_resolve_vllm_executable(), *cmd[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
