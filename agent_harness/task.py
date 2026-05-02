from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from interpretability.artifacts import write_json, write_jsonl, write_text

from .backends import ModelBackend
from .task_actions import apply_edits, run_commands
from .task_checks import check_diagnostics, check_passed, evaluate_check, excerpt, score_checks, score_from_diagnostics
from .task_loading import load_task, prepare_work_dir
from .task_paths import normalize_edit_path, safe_path, work_dir_prefixes
from .task_snapshots import diagnostic_paths, file_snapshots


@dataclass(frozen=True)
class TaskResult:
    functionality: float
    work_dir: Path
    trace: str
    events: list[dict[str, Any]]
    check_diagnostics: list[dict[str, Any]]
    file_snapshots: dict[str, dict[str, Any]]


def run_task(task_path: str | Path, backend: ModelBackend, run_dir: str | Path, timeout_s: int) -> TaskResult:
    run_path = Path(run_dir)
    task_file, task = load_task(task_path)
    work_dir = prepare_work_dir(task_file, task, run_path)

    action = backend.propose_actions(str(task.get("instruction", "")), work_dir)
    events: list[dict[str, Any]] = []
    if action.raw_response is not None:
        write_text(run_path / "raw_response.txt", action.raw_response)
    if action.parsed_action:
        write_json(run_path / "parsed_action.json", action.parsed_action)
    events.extend(apply_edits(work_dir, action))
    events.extend(run_commands(work_dir, action.commands, task.get("allowed_commands", []), timeout_s))

    checks = task.get("checks", [])
    check_diagnostics = _check_diagnostics(work_dir, checks)
    snapshots = _file_snapshots(work_dir, _diagnostic_paths(check_diagnostics))
    score = _score_from_diagnostics(check_diagnostics, float(task.get("max_score", 1.0)))
    write_text(run_path / "decision_trace.md", action.decision_trace, trailing_newline=True)
    write_json(run_path / "check_diagnostics.json", check_diagnostics)
    write_json(run_path / "final_file_snapshots.json", snapshots)
    write_jsonl(run_path / "tool_log.jsonl", events)
    return TaskResult(
        functionality=score,
        work_dir=work_dir,
        trace=action.decision_trace,
        events=events,
        check_diagnostics=check_diagnostics,
        file_snapshots=snapshots,
    )


def _score_checks(work_dir: Path, checks: list[dict[str, Any]], max_score: float) -> float:
    return score_checks(work_dir, checks, max_score)


def _score_from_diagnostics(checks: list[dict[str, Any]], max_score: float) -> float:
    return score_from_diagnostics(checks, max_score)


def _check_passed(work_dir: Path, check: dict[str, Any]) -> bool:
    return check_passed(work_dir, check)


def _check_diagnostics(work_dir: Path, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return check_diagnostics(work_dir, checks)


def _evaluate_check(work_dir: Path, check: dict[str, Any]) -> dict[str, Any]:
    return evaluate_check(work_dir, check)


def _diagnostic_paths(checks: list[dict[str, Any]]) -> list[str]:
    return diagnostic_paths(checks)


def _file_snapshots(work_dir: Path, rel_paths: list[str]) -> dict[str, dict[str, Any]]:
    return file_snapshots(work_dir, rel_paths)


def _excerpt(text: str, needle: str | None = None, limit: int = 240) -> str:
    return excerpt(text, needle, limit)


def _safe_path(root: Path, rel_path: str) -> Path:
    return safe_path(root, rel_path)


def _normalize_edit_path(root: Path, path: str) -> str:
    return normalize_edit_path(root, path)


def _work_dir_prefixes(root: Path) -> list[Path]:
    return work_dir_prefixes(root)
