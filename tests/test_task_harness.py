from __future__ import annotations

from pathlib import Path

from agent_harness.backends import MockBackend
from agent_harness.task import run_task


def test_mock_task_runs(tmp_path: Path) -> None:
    result = run_task("tasks/replace_token.yaml", MockBackend(), tmp_path, timeout_s=10)
    assert result.functionality == 1.0
    assert (tmp_path / "decision_trace.md").exists()
    assert (tmp_path / "tool_log.jsonl").exists()

