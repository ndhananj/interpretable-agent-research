from __future__ import annotations

import json
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
            events.append({"type": "command_denied", "command": " ".join(command)})
            continue
        proc = subprocess.run(command, cwd=work_dir, text=True, capture_output=True, timeout=timeout_s, check=False)
        events.append(
            {
                "type": "command",
                "command": " ".join(command),
                "returncode": proc.returncode,
                "stdout": proc.stdout[-500:],
                "stderr": proc.stderr[-500:],
            }
        )

    score = _score_checks(work_dir, task.get("checks", []), float(task.get("max_score", 1.0)))
    (run_path / "decision_trace.md").write_text(action.decision_trace + "\n", encoding="utf-8")
    with (run_path / "tool_log.jsonl").open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, sort_keys=True) + "\n")
    return TaskResult(functionality=score, work_dir=work_dir, trace=action.decision_trace, events=events)


def _score_checks(work_dir: Path, checks: list[dict[str, Any]], max_score: float) -> float:
    if not checks:
        return 0.0
    passed = sum(1 for check in checks if _check_passed(work_dir, check))
    return max_score * passed / len(checks)


def _check_passed(work_dir: Path, check: dict[str, Any]) -> bool:
    target = _safe_path(work_dir, str(check["path"]))
    text = target.read_text(encoding="utf-8") if target.exists() else ""
    expected = str(check["text"])
    check_type = check["type"]
    return (
        (check_type == "file_contains" and expected in text)
        or (check_type == "file_not_contains" and expected not in text)
    )


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
