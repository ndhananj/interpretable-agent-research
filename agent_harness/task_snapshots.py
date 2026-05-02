from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .task_checks import excerpt
from .task_paths import safe_path


def diagnostic_paths(checks: list[dict[str, Any]]) -> list[str]:
    return sorted({str(check["path"]) for check in checks})


def file_snapshots(work_dir: Path, rel_paths: list[str]) -> dict[str, dict[str, Any]]:
    return {rel_path: file_snapshot(work_dir, rel_path) for rel_path in rel_paths}


def file_snapshot(work_dir: Path, rel_path: str) -> dict[str, Any]:
    target = safe_path(work_dir, rel_path)
    if not target.exists():
        return {"exists": False, "excerpt": "", "sha256": None}
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        data = target.read_bytes()
        return {"exists": True, "excerpt": "(binary file omitted)", "sha256": hashlib.sha256(data).hexdigest()}
    return {"exists": True, "excerpt": excerpt(text), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
