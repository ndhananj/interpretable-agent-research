from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from interpretability.config import load_yaml


def load_task(task_path: str | Path) -> tuple[Path, dict[str, Any]]:
    task_file = Path(task_path)
    return task_file, load_yaml(task_file)


def prepare_work_dir(task_file: Path, task: dict[str, Any], run_dir: Path) -> Path:
    work_dir = run_dir / str(task.get("work_dir_name", "work"))
    fixture = task_file.parent / str(task["fixture_dir"]).replace("tasks/", "")
    if not fixture.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture}")
    shutil.copytree(fixture, work_dir, dirs_exist_ok=True)
    return work_dir
