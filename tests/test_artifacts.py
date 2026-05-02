from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from interpretability.artifacts import read_json, read_jsonl, write_json, write_jsonl


@dataclass(frozen=True)
class Payload:
    value: int


def test_artifact_helpers_round_trip_dataclass_json_and_jsonl(tmp_path: Path) -> None:
    json_path = tmp_path / "payload.json"
    jsonl_path = tmp_path / "events.jsonl"

    write_json(json_path, Payload(3))
    write_jsonl(jsonl_path, [{"b": 2}, {"a": 1}])

    assert read_json(json_path) == {"value": 3}
    assert read_json(tmp_path / "missing.json", {"default": True}) == {"default": True}
    assert read_jsonl(jsonl_path) == [{"b": 2}, {"a": 1}]
