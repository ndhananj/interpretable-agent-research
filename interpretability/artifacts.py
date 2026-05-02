from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def payload(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value


def read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return {} if default is None else default
    return json.loads(target.read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: str | Path, value: Any, *, sort_keys: bool = True) -> None:
    Path(path).write_text(json.dumps(payload(value), indent=2, sort_keys=sort_keys) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(payload(record), sort_keys=True) + "\n")


def write_text(path: str | Path, text: str, *, trailing_newline: bool = False) -> None:
    Path(path).write_text(text + ("\n" if trailing_newline else ""), encoding="utf-8")
