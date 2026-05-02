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
    assert (tmp_path / "decision_trace.md").exists()
    assert (tmp_path / "tool_log.jsonl").exists()


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


def test_task_accepts_absolute_edit_path_inside_work_dir(tmp_path: Path) -> None:
    result = run_task(
        "tasks/replace_token.yaml",
        _EditPathBackend("{work_dir}/input.txt"),
        tmp_path.resolve(),
        timeout_s=10,
    )

    assert result.functionality == 1.0
