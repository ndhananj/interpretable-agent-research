from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from interpretability.artifacts import write_json


@dataclass(frozen=True)
class EvidenceRecord:
    technique: str
    status: str
    score: float
    summary: str
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Recommendation:
    kind: str
    readiness: float
    summary: str
    artifact: str | None = None


@dataclass(frozen=True)
class MechanisticReport:
    version: int
    run_dir: str
    model: dict[str, Any]
    evidence: list[EvidenceRecord]
    recommendations: list[Recommendation]
    scores: dict[str, float]
    legacy: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run_dir": self.run_dir,
            "model": self.model,
            "evidence": [_dataclass_dict(item) for item in self.evidence],
            "recommendations": [_dataclass_dict(item) for item in self.recommendations],
            "scores": dict(self.scores),
            "legacy": dict(self.legacy),
            # Legacy compatibility for existing metrics configs and old reports.
            **self.legacy,
            **self.scores,
        }

    def write(self, path: str | Path) -> None:
        write_json(path, self.to_dict(), sort_keys=False)


def _dataclass_dict(value: Any) -> dict[str, Any]:
    return {key: getattr(value, key) for key in value.__dataclass_fields__}

