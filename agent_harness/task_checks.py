from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .task_paths import safe_path


def score_checks(work_dir: Path, checks: list[dict[str, Any]], max_score: float) -> float:
    return score_from_diagnostics(check_diagnostics(work_dir, checks), max_score)


def score_from_diagnostics(checks: list[dict[str, Any]], max_score: float) -> float:
    if not checks:
        return 0.0
    return max_score * sum(1 for check in checks if check["passed"]) / len(checks)


def check_passed(work_dir: Path, check: dict[str, Any]) -> bool:
    return evaluate_check(work_dir, check)["passed"]


def check_diagnostics(work_dir: Path, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [evaluate_check(work_dir, check) for check in checks]


def evaluate_check(work_dir: Path, check: dict[str, Any]) -> dict[str, Any]:
    target = safe_path(work_dir, str(check["path"]))
    raw = target.read_bytes() if target.exists() else b""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = ""
    expected = str(check["text"])
    check_type = check["type"]
    evaluators = {
        "file_contains": lambda: expected in text,
        "file_not_contains": lambda: expected not in text,
    }
    passed = evaluators.get(check_type, lambda: False)()
    return {
        "type": check_type,
        "path": str(check["path"]),
        "expected_text": expected,
        "passed": passed,
        "final_excerpt": excerpt(text, expected),
        "final_sha256": hashlib.sha256(raw).hexdigest(),
    }


def excerpt(text: str, needle: str | None = None, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    if needle:
        index = text.find(needle)
        if index >= 0:
            start = max(0, index - limit // 2)
            end = min(len(text), start + limit)
            return ("..." if start else "") + text[start:end] + ("..." if end < len(text) else "")
    return text[: limit - 3] + "..."
