from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentAction:
    decision_trace: str
    edits: dict[str, str]
    commands: list[list[str]]


class ModelBackend:
    def propose_actions(self, instruction: str, work_dir: Path) -> AgentAction:
        raise NotImplementedError


class MockBackend(ModelBackend):
    """Deterministic backend for offline smoke tests."""

    def propose_actions(self, instruction: str, work_dir: Path) -> AgentAction:
        edits: dict[str, str] = {}
        input_path = work_dir / "input.txt"
        if input_path.exists():
            text = input_path.read_text(encoding="utf-8")
            edits["input.txt"] = text.replace("TODO", "DONE")
        return AgentAction(
            decision_trace=(
                "I read the task file, edit input.txt to replace the target token, "
                "and test the result with a small python3 command."
            ),
            edits=edits,
            commands=[["python3", "-c", "from pathlib import Path; assert 'DONE' in Path('input.txt').read_text()"]],
        )


def make_backend(config: dict) -> ModelBackend:
    backend = str(config.get("backend", "mock"))
    if backend == "mock":
        return MockBackend()
    raise NotImplementedError(
        f"Backend {backend!r} is not implemented in v1. Use 'mock' for offline runs."
    )

