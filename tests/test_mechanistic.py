from __future__ import annotations

import json
from pathlib import Path

from interpretability.artifacts import read_json
from interpretability.mechanistic import EvidenceRecord, build_report, write_mechanistic_report


def test_mechanistic_report_serializes_scores_and_legacy_keys(tmp_path: Path) -> None:
    report = build_report(
        tmp_path,
        {"name": "base", "api_key": "secret"},
        [
            EvidenceRecord(
                technique="synthetic",
                status="available",
                score=0.8,
                summary="synthetic evidence",
                metrics={"feature_sparsity": 0.8, "attribution_stability": 0.7},
            )
        ],
    )

    payload = report.to_dict()

    assert payload["model"] == {"name": "base"}
    assert payload["scores"]["feature_sparsity"] == 0.8
    assert payload["scores"]["attribution_stability"] == 0.7
    assert payload["sparsity"] == 0.8
    assert payload["stability"] == 0.7


def test_write_mechanistic_report_falls_back_and_exports_specs(tmp_path: Path) -> None:
    report = write_mechanistic_report(tmp_path, {"fallback": {"missing_backend_default": 0.25}}, {"name": "base"})
    payload = read_json(tmp_path / "mechanistic.json")

    assert len(report.evidence) == 7
    assert payload["scores"]["adapter_locality"] == 0.25
    assert payload["scores"]["pruning_retained_functionality"] == 0.9
    assert (tmp_path / "pruning_mask_spec.json").exists()
    assert (tmp_path / "sparse_training_config.json").exists()


def test_lora_geometry_reads_adapter_metadata(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"target_modules": ["q_proj", "v_proj"]}),
        encoding="utf-8",
    )

    report = write_mechanistic_report(tmp_path / "run", {}, {"adapter_path": str(adapter)})

    lora = next(item for item in report.evidence if item.technique == "lora_delta_geometry")
    assert lora.status == "available"
    assert lora.metrics["adapter_locality"] > 0.0
    assert lora.metadata["target_modules"] == ["q_proj", "v_proj"]

