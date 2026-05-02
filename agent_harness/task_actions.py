from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from agent_harness.backends import AgentAction

from .task_paths import normalize_edit_path, safe_path


def apply_edits(work_dir: Path, action: AgentAction) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for rel_path, content in action.edits.items():
        normalized_path = normalize_edit_path(work_dir, rel_path)
        target = safe_path(work_dir, normalized_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        events.append({"type": "edit", "path": normalized_path})
    return events


def run_commands(
    work_dir: Path,
    commands: list[list[str]],
    allowed_commands: list[Any],
    timeout_s: int,
) -> list[dict[str, Any]]:
    allowed = {str(command) for command in allowed_commands}
    events: list[dict[str, Any]] = []
    for command in commands:
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
    return events
