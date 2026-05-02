from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from interpretability.config import load_yaml


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a configured LoRA adapter through local vLLM")
    parser.add_argument("--config", default="configs/vllm.yaml")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--dry-run", action="store_true")
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
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
