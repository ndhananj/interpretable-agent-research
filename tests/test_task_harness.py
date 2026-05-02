from __future__ import annotations

from pathlib import Path

from agent_harness.backends import AgentAction, MockBackend, ModelBackend
from agent_harness.task import run_task


class _EditPathBackend(ModelBackend):
    def __init__(self, path_template: str) -> None:
        self.path_template = path_template

    def propose_actions(self, instruction: str, work_dir: Path) -> AgentAction:
        return AgentAction(
            decision_trace="edit with a model-provided path",
            edits={self.path_template.format(work_dir=work_dir): "DONE\n"},
            commands=[],
        )


def test_mock_task_runs(tmp_path: Path) -> None:
    result = run_task("tasks/replace_token.yaml", MockBackend(), tmp_path, timeout_s=10)
    assert result.functionality == 1.0
    assert result.check_diagnostics[0]["type"] == "file_contains"
    assert result.check_diagnostics[0]["path"] == "input.txt"
    assert result.check_diagnostics[0]["expected_text"] == "DONE"
    assert result.check_diagnostics[0]["passed"] is True
    assert result.file_snapshots["input.txt"]["exists"] is True
    assert result.file_snapshots["input.txt"]["sha256"]
    assert (tmp_path / "decision_trace.md").exists()
    assert (tmp_path / "tool_log.jsonl").exists()
    assert (tmp_path / "check_diagnostics.json").exists()
    assert (tmp_path / "final_file_snapshots.json").exists()


def test_task_normalizes_model_edit_path_with_work_dir_prefix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_path = Path("tasks/replace_token.yaml").resolve()
    monkeypatch.chdir(tmp_path)
    run_dir = Path("runs/model-compare/base/replace_token")

    result = run_task(
        task_path,
        _EditPathBackend("{work_dir}/input.txt"),
        run_dir,
        timeout_s=10,
    )

    assert result.functionality == 1.0
    assert (run_dir / "work" / "input.txt").read_text(encoding="utf-8") == "DONE\n"
    assert '"path": "input.txt"' in (run_dir / "tool_log.jsonl").read_text(encoding="utf-8")


def test_task_records_command_argv_and_return_code(tmp_path: Path) -> None:
    result = run_task("tasks/replace_token.yaml", MockBackend(), tmp_path, timeout_s=10)

    command_events = [event for event in result.events if event["type"] == "command"]
    assert command_events[0]["argv"][0] == "python3"
    assert command_events[0]["returncode"] == 0


def test_task_accepts_absolute_edit_path_inside_work_dir(tmp_path: Path) -> None:
    result = run_task(
        "tasks/replace_token.yaml",
        _EditPathBackend("{work_dir}/input.txt"),
        tmp_path.resolve(),
        timeout_s=10,
    )

    assert result.functionality == 1.0


class _CommandBackend(ModelBackend):
    def __init__(self, commands: list[list[str]]) -> None:
        self.commands = commands

    def propose_actions(self, instruction: str, work_dir: Path) -> AgentAction:
        return AgentAction(decision_trace="try command", edits={}, commands=self.commands)


def test_task_records_denied_command(tmp_path: Path) -> None:
    result = run_task("tasks/replace_token.yaml", _CommandBackend([["sh", "-c", "echo denied"]]), tmp_path, timeout_s=10)

    assert result.events[0]["type"] == "command_denied"
    assert result.events[0]["argv"] == ["sh", "-c", "echo denied"]


def test_task_missing_fixture_fails_clearly(tmp_path: Path) -> None:
    task_path = tmp_path / "missing_fixture.yaml"
    task_path.write_text(
        """
fixture_dir: does_not_exist
checks: []
""",
        encoding="utf-8",
    )

    try:
        run_task(task_path, MockBackend(), tmp_path / "run", timeout_s=10)
    except FileNotFoundError as exc:
        assert "Fixture not found" in str(exc)
    else:
        raise AssertionError("missing fixture should fail")


def test_task_snapshot_handles_binary_file(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "blob.bin").write_bytes(b"\xff\x00\xfe")
    task_path = tmp_path / "binary_task.yaml"
    task_path.write_text(
        f"""
fixture_dir: {fixture}
checks:
  - type: file_contains
    path: blob.bin
    text: anything
""",
        encoding="utf-8",
    )

    result = run_task(task_path, _CommandBackend([]), tmp_path / "run", timeout_s=10)

    assert result.file_snapshots["blob.bin"]["excerpt"] == "(binary file omitted)"
    assert result.file_snapshots["blob.bin"]["sha256"]
