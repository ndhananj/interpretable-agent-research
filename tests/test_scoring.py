from __future__ import annotations

import json
from pathlib import Path

from interpretability.scoring import score_run


def test_functionality_floor_rejects(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text(json.dumps({"functionality": 0.5}), encoding="utf-8")
    (tmp_path / "decision_trace.md").write_text("read edit test command input.txt", encoding="utf-8")
    (tmp_path / "tool_log.jsonl").write_text(json.dumps({"type": "edit", "path": "input.txt"}) + "\n", encoding="utf-8")
    cfg = {"weights": {"behavior_trace_faithfulness": 1.0}, "behavior": {"required_terms": ["read", "edit", "test"]}}
    result = score_run(tmp_path, cfg, baseline_functionality=1.0, floor_ratio=0.9)
    assert not result.accepted
    assert result.reason == "functionality floor failed"


def test_explainability_accepts_when_floor_passes(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text(json.dumps({"functionality": 1.0}), encoding="utf-8")
    (tmp_path / "decision_trace.md").write_text("read edit test command input.txt", encoding="utf-8")
    (tmp_path / "tool_log.jsonl").write_text(json.dumps({"type": "edit", "path": "input.txt"}) + "\n", encoding="utf-8")
    cfg = {"weights": {"behavior_trace_faithfulness": 1.0}, "behavior": {"required_terms": ["read", "edit", "test"]}}
    result = score_run(tmp_path, cfg, baseline_functionality=1.0, floor_ratio=0.9)
    assert result.accepted

