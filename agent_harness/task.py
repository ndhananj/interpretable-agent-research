from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from interpretability.config import load_yaml

from .backends import ModelBackend


@dataclass(frozen=True)
class TaskResult:
    functionality: float
    work_dir: Path
    trace: str
    events: list[dict[str, Any]]
    check_diagnostics: list[dict[str, Any]]
    file_snapshots: dict[str, dict[str, Any]]


def run_task(task_path: str | Path, backend: ModelBackend, run_dir: str | Path, timeout_s: int) -> TaskResult:
    task_file = Path(task_path)
    task = load_yaml(task_file)
    run_path = Path(run_dir)
    work_dir = run_path / str(task.get("work_dir_name", "work"))
    fixture = task_file.parent / str(task["fixture_dir"]).replace("tasks/", "")
    if fixture.exists():
        shutil.copytree(fixture, work_dir, dirs_exist_ok=True)
    else:
        raise FileNotFoundError(f"Fixture not found: {fixture}")

    action = backend.propose_actions(str(task.get("instruction", "")), work_dir)
    events: list[dict[str, Any]] = []
    if action.raw_response is not None:
        (run_path / "raw_response.txt").write_text(action.raw_response, encoding="utf-8")
    if action.parsed_action:
        (run_path / "parsed_action.json").write_text(
            json.dumps(action.parsed_action, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for rel_path, content in action.edits.items():
        normalized_path = _normalize_edit_path(work_dir, rel_path)
        target = _safe_path(work_dir, normalized_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        events.append({"type": "edit", "path": normalized_path})

    allowed = {str(cmd) for cmd in task.get("allowed_commands", [])}
    for command in action.commands:
        if not command:
            continue
        if command[0] not in allowed:
            events.append({"type": "command_denied", "command": " ".join(command), "argv": command})
            continue
        proc = subprocess.run(command, cwd=work_dir, text=True, capture_output=True, timeout=timeout_s, check=False)
        events.append(
            {
                "type": "command",
                "command": " ".join(command),
                "argv": command,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-500:],
                "stderr": proc.stderr[-500:],
            }
        )

    checks = task.get("checks", [])
    check_diagnostics = _check_diagnostics(work_dir, checks)
    snapshots = _file_snapshots(work_dir, _diagnostic_paths(check_diagnostics))
    score = _score_from_diagnostics(check_diagnostics, float(task.get("max_score", 1.0)))
    (run_path / "decision_trace.md").write_text(action.decision_trace + "\n", encoding="utf-8")
    (run_path / "check_diagnostics.json").write_text(
        json.dumps(check_diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_path / "final_file_snapshots.json").write_text(
        json.dumps(snapshots, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (run_path / "tool_log.jsonl").open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, sort_keys=True) + "\n")
    return TaskResult(
        functionality=score,
        work_dir=work_dir,
        trace=action.decision_trace,
        events=events,
        check_diagnostics=check_diagnostics,
        file_snapshots=snapshots,
    )


def _score_checks(work_dir: Path, checks: list[dict[str, Any]], max_score: float) -> float:
    return _score_from_diagnostics(_check_diagnostics(work_dir, checks), max_score)


def _score_from_diagnostics(checks: list[dict[str, Any]], max_score: float) -> float:
    if not checks:
        return 0.0
    passed = sum(1 for check in checks if check["passed"])
    return max_score * passed / len(checks)


def _check_passed(work_dir: Path, check: dict[str, Any]) -> bool:
    return _evaluate_check(work_dir, check)["passed"]


def _check_diagnostics(work_dir: Path, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_evaluate_check(work_dir, check) for check in checks]


def _evaluate_check(work_dir: Path, check: dict[str, Any]) -> dict[str, Any]:
    target = _safe_path(work_dir, str(check["path"]))
    text = target.read_text(encoding="utf-8") if target.exists() else ""
    expected = str(check["text"])
    check_type = check["type"]
    passed = (
        (check_type == "file_contains" and expected in text)
        or (check_type == "file_not_contains" and expected not in text)
    )
    return {
        "type": check_type,
        "path": str(check["path"]),
        "expected_text": expected,
        "passed": passed,
        "final_excerpt": _excerpt(text, expected),
        "final_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _diagnostic_paths(checks: list[dict[str, Any]]) -> list[str]:
    return sorted({str(check["path"]) for check in checks})


def _file_snapshots(work_dir: Path, rel_paths: list[str]) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for rel_path in rel_paths:
        target = _safe_path(work_dir, rel_path)
        if not target.exists():
            snapshots[rel_path] = {"exists": False, "excerpt": "", "sha256": None}
            continue
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            data = target.read_bytes()
            snapshots[rel_path] = {
                "exists": True,
                "excerpt": "(binary file omitted)",
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            continue
        snapshots[rel_path] = {
            "exists": True,
            "excerpt": _excerpt(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    return snapshots


def _excerpt(text: str, needle: str | None = None, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    if needle:
        index = text.find(needle)
        if index >= 0:
            start = max(0, index - limit // 2)
            end = min(len(text), start + limit)
            prefix = "..." if start else ""
            suffix = "..." if end < len(text) else ""
            return prefix + text[start:end] + suffix
    return text[: limit - 3] + "..."


def _safe_path(root: Path, rel_path: str) -> Path:
    target = (root / rel_path).resolve()
    root_resolved = root.resolve()
    if root_resolved not in target.parents and target != root_resolved:
        raise ValueError(f"Path escapes work dir: {rel_path}")
    return target


def _normalize_edit_path(root: Path, path: str) -> str:
    raw = Path(path)
    root_resolved = root.resolve()
    if raw.is_absolute():
        raw_resolved = raw.resolve()
        if root_resolved not in raw_resolved.parents and raw_resolved != root_resolved:
            raise ValueError(f"Path escapes work dir: {path}")
        return raw_resolved.relative_to(root_resolved).as_posix()

    raw_posix = raw.as_posix()
    for prefix in _work_dir_prefixes(root):
        prefix_posix = prefix.as_posix().rstrip("/")
        if raw_posix == prefix_posix:
            return "."
        if raw_posix.startswith(f"{prefix_posix}/"):
            return raw_posix[len(prefix_posix) + 1 :]
    return raw_posix


def _work_dir_prefixes(root: Path) -> list[Path]:
    prefixes = [root, root.resolve()]
    try:
        prefixes.append(root.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        pass
    return prefixes
