from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScoreResult:
    functionality: float
    explainability: float
    accepted: bool
    reason: str
    details: dict[str, float]


def score_run(
    run_dir: str | Path,
    metrics_config: dict[str, Any],
    baseline_functionality: float,
    floor_ratio: float,
    incumbent_explainability: float = 0.0,
) -> ScoreResult:
    run_path = Path(run_dir)
    metrics = _read_json(run_path / "metrics.json")
    functionality = float(metrics.get("functionality", 0.0))
    floor = baseline_functionality * floor_ratio
    details = explainability_details(
        tool_log_path=run_path / "tool_log.jsonl",
        trace_path=run_path / "decision_trace.md",
        mechanistic_path=run_path / "mechanistic.json",
        metrics_config=metrics_config,
    )
    explainability = sum(details.values())
    if functionality < floor:
        return ScoreResult(functionality, explainability, False, "functionality floor failed", details)
    if explainability <= incumbent_explainability:
        return ScoreResult(functionality, explainability, False, "explainability did not improve", details)
    return ScoreResult(functionality, explainability, True, "accepted", details)


def explainability_details(
    tool_log_path: Path,
    trace_path: Path,
    mechanistic_path: Path,
    metrics_config: dict[str, Any],
) -> dict[str, float]:
    weights = metrics_config.get("weights", {})
    behavior_cfg = metrics_config.get("behavior", {})
    mech_cfg = metrics_config.get("mechanistic", {})
    trace = trace_path.read_text(encoding="utf-8") if trace_path.exists() else ""
    log_events = _read_jsonl(tool_log_path)
    mechanistic = _read_json(mechanistic_path) if mechanistic_path.exists() else {}

    faithfulness = _faithfulness(trace, log_events, behavior_cfg)
    concision = _concision(trace, int(behavior_cfg.get("max_trace_words", 180)))
    alignment = _tool_log_alignment(trace, log_events)
    sparsity = float(mechanistic.get("sparsity", mech_cfg.get("missing_backend_default", 0.30)))
    stability = float(mechanistic.get("stability", mech_cfg.get("missing_backend_default", 0.30)))

    return {
        "behavior_trace_faithfulness": faithfulness * float(weights.get("behavior_trace_faithfulness", 0.0)),
        "behavior_trace_concision": concision * float(weights.get("behavior_trace_concision", 0.0)),
        "tool_log_alignment": alignment * float(weights.get("tool_log_alignment", 0.0)),
        "mechanistic_sparsity": max(0.0, min(1.0, sparsity)) * float(weights.get("mechanistic_sparsity", 0.0)),
        "mechanistic_stability": max(0.0, min(1.0, stability)) * float(weights.get("mechanistic_stability", 0.0)),
    }


def _faithfulness(trace: str, events: list[dict[str, Any]], cfg: dict[str, Any]) -> float:
    if not trace.strip() or not events:
        return 0.0
    required = [str(term).lower() for term in cfg.get("required_terms", [])]
    lowered = trace.lower()
    required_score = sum(1 for term in required if term in lowered) / max(len(required), 1)
    event_terms = {str(event.get("type", "")).lower() for event in events}
    event_score = sum(1 for term in event_terms if term and term in lowered) / max(len(event_terms), 1)
    return (required_score + event_score) / 2


def _concision(trace: str, max_words: int) -> float:
    words = trace.split()
    if not words:
        return 0.0
    if len(words) <= max_words:
        return 1.0
    return max(0.0, max_words / len(words))


def _tool_log_alignment(trace: str, events: list[dict[str, Any]]) -> float:
    if not events:
        return 0.0
    lowered = trace.lower()
    matched = 0
    for event in events:
        command = str(event.get("command", "")).lower()
        path = str(event.get("path", "")).lower()
        if command and command.split()[0] in lowered:
            matched += 1
        elif path and Path(path).name.lower() in lowered:
            matched += 1
    return matched / len(events)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events

