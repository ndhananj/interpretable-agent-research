from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from importlib.util import find_spec

import psutil


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_count: int
    load_1m: float
    load_ratio: float
    total_memory_gib: float
    available_memory_gib: float
    cuda_available: bool
    cuda_device_count: int
    nvidia_smi_available: bool
    vllm_available: bool


def detect_resources() -> ResourceSnapshot:
    cpu_count = os.cpu_count() or 1
    load_1m = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    mem = psutil.virtual_memory()
    cuda_available = False
    cuda_device_count = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torch

            cuda_available = bool(torch.cuda.is_available())
            cuda_device_count = int(torch.cuda.device_count())
    except Exception:
        cuda_available = False
        cuda_device_count = 0
    nvidia_smi_available = _nvidia_smi_works()
    vllm_available = _module_available("vllm")
    return ResourceSnapshot(
        cpu_count=cpu_count,
        load_1m=load_1m,
        load_ratio=load_1m / max(cpu_count, 1),
        total_memory_gib=mem.total / 1024**3,
        available_memory_gib=mem.available / 1024**3,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        nvidia_smi_available=nvidia_smi_available,
        vllm_available=vllm_available,
    )


def should_backoff(config: dict, snapshot: ResourceSnapshot | None = None) -> tuple[bool, str]:
    snapshot = snapshot or detect_resources()
    max_load_ratio = float(config.get("max_load_ratio", 0.75))
    min_mem = float(config.get("min_available_memory_gib", 8))
    if snapshot.load_ratio > max_load_ratio:
        return True, f"load ratio {snapshot.load_ratio:.2f} exceeds {max_load_ratio:.2f}"
    if snapshot.available_memory_gib < min_mem:
        return True, f"available memory {snapshot.available_memory_gib:.1f} GiB below {min_mem:.1f} GiB"
    return False, "resources available"


def configure_conservative_threads(max_threads: int) -> None:
    threads = max(1, int(max_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:
        pass


def _module_available(name: str) -> bool:
    return find_spec(name) is not None


def _nvidia_smi_works() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False
