from __future__ import annotations

import argparse
import importlib
import importlib.metadata
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
class DependencyCompatibilityError:
    reason: str
    repair_command: str


@dataclass(frozen=True)
class CudaPreflightResult:
    ok: bool
    python_executable: str = "unknown"
    torch_version: str = "unknown"
    torch_cuda: str = "unknown"
    cuda_available: bool = False
    device_count: int | None = None
    nvidia_smi: str | None = None
    vllm_version: str = "unknown"
    transformers_version: str = "unknown"
    dependency_error: DependencyCompatibilityError | None = None
    vllm_native_import: str | None = None
    error: str | None = None


DEPENDENCY_PACKAGES = ("vllm", "transformers")
VLLM_066_POST1_TRANSFORMERS_SPEC = "transformers>=4.56.2,<5"


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
    model = validate_vllm_model_config(config)
    base_model = model["name"]
    adapter_name = model["adapter_name"]
    adapter_path = model["adapter_path"]
    base_url = model["base_url"]

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


def validate_vllm_model_config(config: dict[str, Any]) -> dict[str, str]:
    model = config.get("model")
    if not isinstance(model, dict):
        raise SystemExit("Config must include a model mapping")
    required = (
        ("name", "model.name must be a non-empty base model name"),
        ("adapter_name", "model.adapter_name must be a non-empty LoRA module name"),
        ("adapter_path", "model.adapter_path must be a non-empty path to a LoRA adapter"),
        ("base_url", "model.base_url must be a non-empty URL"),
    )
    if model.get("backend") != "vllm_openai":
        raise SystemExit("model.backend must be vllm_openai")
    values: dict[str, str] = {}
    for key, message in required:
        value = model.get(key)
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(message)
        values[key] = value
    return values


def _resolve_adapter_path(adapter_path: str) -> Path:
    path = Path(adapter_path).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def run_cuda_preflight() -> CudaPreflightResult:
    python_executable = sys.executable
    package_versions = get_installed_dependency_versions()
    vllm_version = package_versions["vllm"]
    transformers_version = package_versions["transformers"]
    dependency_error = evaluate_dependency_compatibility(package_versions)
    if dependency_error is not None:
        return CudaPreflightResult(
            ok=False,
            python_executable=python_executable,
            vllm_version=vllm_version,
            transformers_version=transformers_version,
            dependency_error=dependency_error,
            error="Python package dependency mismatch",
        )

    nvidia_smi = _run_nvidia_smi_list()
    vllm_native_import = _run_vllm_native_import()

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return CudaPreflightResult(
            ok=False,
            python_executable=python_executable,
            nvidia_smi=nvidia_smi,
            vllm_version=vllm_version,
            transformers_version=transformers_version,
            vllm_native_import=vllm_native_import,
            error=f"torch import failed: {exc}",
        )

    torch_version = str(getattr(torch, "__version__", "unknown"))
    version = getattr(torch, "version", None)
    torch_cuda_value = getattr(version, "cuda", None) if version is not None else None
    torch_cuda = str(torch_cuda_value or "not available")
    if not torch_cuda_value:
        return CudaPreflightResult(
            ok=False,
            python_executable=python_executable,
            torch_version=torch_version,
            torch_cuda=torch_cuda,
            nvidia_smi=nvidia_smi,
            vllm_version=vllm_version,
            transformers_version=transformers_version,
            vllm_native_import=vllm_native_import,
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
            python_executable=python_executable,
            torch_version=torch_version,
            torch_cuda=torch_cuda,
            nvidia_smi=nvidia_smi,
            vllm_version=vllm_version,
            transformers_version=transformers_version,
            vllm_native_import=vllm_native_import,
            error=f"torch.cuda.is_available() failed: {exc}",
        )

    device_count: int | None = None
    if cuda_available:
        try:
            device_count = int(torch.cuda.device_count())
        except Exception as exc:
            return CudaPreflightResult(
                ok=False,
                python_executable=python_executable,
                torch_version=torch_version,
                torch_cuda=torch_cuda,
                cuda_available=cuda_available,
                nvidia_smi=nvidia_smi,
                vllm_version=vllm_version,
                transformers_version=transformers_version,
                vllm_native_import=vllm_native_import,
                error=f"torch.cuda.device_count() failed: {exc}",
            )

    if not cuda_available:
        error = "torch reports CUDA is unavailable"
        if cuda_warning:
            error = f"{error}: {cuda_warning}"
        return CudaPreflightResult(
            ok=False,
            python_executable=python_executable,
            torch_version=torch_version,
            torch_cuda=torch_cuda,
            cuda_available=False,
            device_count=device_count,
            nvidia_smi=nvidia_smi,
            vllm_version=vllm_version,
            transformers_version=transformers_version,
            vllm_native_import=vllm_native_import,
            error=error,
        )
    if device_count is None or device_count < 1:
        return CudaPreflightResult(
            ok=False,
            python_executable=python_executable,
            torch_version=torch_version,
            torch_cuda=torch_cuda,
            cuda_available=True,
            device_count=device_count,
            nvidia_smi=nvidia_smi,
            vllm_version=vllm_version,
            transformers_version=transformers_version,
            vllm_native_import=vllm_native_import,
            error="torch reports zero visible CUDA devices",
        )
    if nvidia_smi is not None and nvidia_smi.startswith("failed:"):
        return CudaPreflightResult(
            ok=False,
            python_executable=python_executable,
            torch_version=torch_version,
            torch_cuda=torch_cuda,
            cuda_available=True,
            device_count=device_count,
            nvidia_smi=nvidia_smi,
            vllm_version=vllm_version,
            transformers_version=transformers_version,
            vllm_native_import=vllm_native_import,
            error="nvidia-smi -L failed",
        )
    if vllm_native_import is not None:
        return CudaPreflightResult(
            ok=False,
            python_executable=python_executable,
            torch_version=torch_version,
            torch_cuda=torch_cuda,
            cuda_available=True,
            device_count=device_count,
            nvidia_smi=nvidia_smi,
            vllm_version=vllm_version,
            transformers_version=transformers_version,
            vllm_native_import=vllm_native_import,
            error="vLLM native extension import failed",
        )
    return CudaPreflightResult(
        ok=True,
        python_executable=python_executable,
        torch_version=torch_version,
        torch_cuda=torch_cuda,
        cuda_available=True,
        device_count=device_count,
        nvidia_smi=nvidia_smi,
        vllm_version=vllm_version,
        transformers_version=transformers_version,
    )


def get_installed_dependency_versions(packages: tuple[str, ...] = DEPENDENCY_PACKAGES) -> dict[str, str]:
    return {package: _get_installed_package_version(package) for package in packages}


def evaluate_dependency_compatibility(package_versions: dict[str, str]) -> DependencyCompatibilityError | None:
    vllm_version = package_versions.get("vllm", "unknown")
    transformers_version = package_versions.get("transformers", "unknown")
    if _version_equals(vllm_version, "0.6.6.post1") and not _version_less_than(transformers_version, "5"):
        return DependencyCompatibilityError(
            reason=(
                "vllm==0.6.6.post1 requires transformers<5; "
                f"installed transformers is {transformers_version}"
            ),
            repair_command=format_dependency_repair_command(VLLM_066_POST1_TRANSFORMERS_SPEC),
        )
    return None


def format_dependency_repair_command(requirement: str) -> str:
    return f"uv pip install {shlex.quote(requirement)}"


def _version_equals(installed: str, expected: str) -> bool:
    return _version_key(installed) == _version_key(expected)


def _version_less_than(installed: str, upper_bound: str) -> bool:
    installed_key = _version_key(installed)
    upper_bound_key = _version_key(upper_bound)
    return installed_key is not None and upper_bound_key is not None and installed_key < upper_bound_key


def _version_key(version: str) -> tuple[int, ...] | None:
    head = version.split("+", 1)[0]
    parts: list[int] = []
    for part in head.replace(".post", ".").split("."):
        if not part:
            continue
        if not part.isdigit():
            return None
        parts.append(int(part))
    return tuple(parts) if parts else None


def _get_installed_package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"
    except Exception as exc:
        return f"unknown ({exc})"


def _run_vllm_native_import() -> str | None:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", "import vllm._C"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return f"failed to run native import check: {exc}"
    if completed.returncode == 0:
        return None
    output = "\n".join(part.strip() for part in (completed.stderr, completed.stdout) if part and part.strip())
    return _summarize_import_failure(output, completed.returncode)


def _summarize_import_failure(output: str, returncode: int) -> str:
    if not output:
        return f"import vllm._C failed with exit code {returncode}"
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped.startswith(("ImportError:", "ModuleNotFoundError:", "OSError:")):
            return f"import vllm._C failed: {stripped}"
    return f"import vllm._C failed: {output.splitlines()[-1].strip()}"


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
        f"python: {result.python_executable}",
        f"torch: {result.torch_version}",
        f"torch.version.cuda: {result.torch_cuda}",
        f"torch.cuda.is_available(): {result.cuda_available}",
        f"vLLM: {result.vllm_version}",
        f"transformers: {result.transformers_version}",
    ]
    if result.device_count is not None:
        lines.append(f"torch.cuda.device_count(): {result.device_count}")
    if result.nvidia_smi is not None:
        lines.append(f"nvidia-smi -L: {result.nvidia_smi}")
    if result.vllm_native_import is not None:
        lines.append(f"vLLM native import: {result.vllm_native_import}")
    if result.error:
        lines.append(f"reason: {result.error}")
    if result.dependency_error is not None:
        lines.extend(
            [
                f"dependency: {result.dependency_error.reason}",
                "",
                "Repair the active vLLM environment without retraining or changing adapters/latest:",
                f"  {result.dependency_error.repair_command}",
                "  python serve_vllm_adapter.py --config configs/vllm.yaml",
                "",
                "Use --skip-preflight only when debugging raw vLLM startup behavior.",
            ]
        )
        return "\n".join(lines)
    if result.vllm_native_import and "libcudart.so." in result.vllm_native_import:
        lines.extend(
            [
                "",
                "The active vLLM extension cannot find the CUDA runtime it was built against.",
                "Fix the NVIDIA driver first if `nvidia-smi -L` fails, then reinstall a matching vLLM/PyTorch CUDA stack.",
            ]
        )
    lines.extend(
        [
            "",
            "Recommended repair order:",
            "  1. Fix or reload the NVIDIA driver until `nvidia-smi` works.",
            "  2. Rebuild or repair the active vLLM environment:",
            "     pip uninstall -y vllm torch torchvision torchaudio 'nvidia-*' cuda-toolkit cuda-bindings cuda-python",
            "     uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "     uv pip install vllm==0.6.6.post1 --torch-backend=cu121",
            "  3. Verify:",
            "     python -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\"",
            "     python -c \"import vllm._C\"",
            "     python serve_vllm_adapter.py --config configs/vllm.yaml",
            "",
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
