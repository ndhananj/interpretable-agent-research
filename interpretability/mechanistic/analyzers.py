from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from interpretability.mechanistic.schema import EvidenceRecord


def lora_delta_geometry(run_dir: Path, config: dict[str, Any], model_config: dict[str, Any]) -> EvidenceRecord:
    adapter_path = _adapter_path(model_config)
    if adapter_path is None or not adapter_path.exists():
        return EvidenceRecord(
            technique="lora_delta_geometry",
            status="unavailable",
            score=_fallback_score(config),
            summary="No local LoRA adapter path was available for CPU-safe geometry analysis.",
            metrics={"adapter_locality": _fallback_score(config), "rank_concentration": 0.0},
            metadata={"adapter_path": str(adapter_path) if adapter_path else None},
        )

    files = [path for path in adapter_path.rglob("*") if path.is_file()]
    named_targets = _adapter_target_modules(adapter_path)
    locality = min(1.0, len(named_targets) / max(len(files), 1)) if named_targets else min(1.0, len(files) / 20.0)
    return EvidenceRecord(
        technique="lora_delta_geometry",
        status="available",
        score=locality,
        summary="Adapter files were found; target-module locality was estimated from adapter metadata.",
        metrics={"adapter_locality": locality, "rank_concentration": min(1.0, len(named_targets) / 16.0)},
        metadata={"adapter_path": str(adapter_path), "target_modules": named_targets, "file_count": len(files)},
    )


def activation_sae(config: dict[str, Any]) -> EvidenceRecord:
    return _optional_record(
        "activation_sparse_autoencoder",
        config,
        "Activation capture or SAE backend is not configured; sparse feature map is a proposal artifact only.",
        {"feature_sparsity": _fallback_score(config), "reconstruction_quality": 0.0},
    )


def circuit_tracing(config: dict[str, Any]) -> EvidenceRecord:
    return _optional_record(
        "circuit_tracing",
        config,
        "Circuit tracing backend is not configured; attribution graph export is a placeholder spec.",
        {"circuit_compactness": _fallback_score(config), "attribution_stability": _fallback_score(config)},
    )


def transcoder_candidates(config: dict[str, Any]) -> EvidenceRecord:
    return _optional_record(
        "transcoder_candidates",
        config,
        "No MLP activation dataset is configured; transcoder replacement spec is not trainable yet.",
        {"replacement_readiness": _fallback_score(config)},
    )


def sae_assisted_circuits(config: dict[str, Any]) -> EvidenceRecord:
    return _optional_record(
        "sae_assisted_circuit_discovery",
        config,
        "SAE feature activations are unavailable; feature-level circuit nodes are deferred.",
        {"feature_sparsity": _fallback_score(config), "circuit_compactness": _fallback_score(config)},
    )


def sparse_training_plan(run_dir: Path, config: dict[str, Any]) -> EvidenceRecord:
    artifact = run_dir / "sparse_training_config.json"
    payload = {
        "objective": "distill_or_retrain_base_model_with_sparse_regularization",
        "max_steps": int(config.get("budgets", {}).get("sparse_training_steps", 0)),
        "target_sparsity": float(config.get("targets", {}).get("weight_sparsity", 0.5)),
        "requires_review": True,
    }
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    score = _fallback_score(config)
    return EvidenceRecord(
        technique="weight_sparse_base_model_plan",
        status="planned",
        score=score,
        summary="Sparse base-model training is exported as a reviewable config, not executed automatically.",
        metrics={"replacement_readiness": score},
        artifacts={"sparse_training_config": artifact.name},
    )


def pruning_masks(run_dir: Path, config: dict[str, Any]) -> EvidenceRecord:
    artifact = run_dir / "pruning_mask_spec.json"
    retained = float(config.get("targets", {}).get("pruning_retained_functionality", 0.9))
    payload = {
        "mask_type": "task_specific_weight_or_node_mask",
        "retained_functionality_target": retained,
        "source": "mechanistic_report_metadata",
        "requires_validation": True,
    }
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return EvidenceRecord(
        technique="pruning_masks",
        status="planned",
        score=retained,
        summary="Pruning mask metadata was exported for later validation against task behavior.",
        metrics={"pruning_retained_functionality": retained, "adapter_locality": _fallback_score(config)},
        artifacts={"pruning_mask_spec": artifact.name},
    )


def _optional_record(technique: str, config: dict[str, Any], summary: str, metrics: dict[str, float]) -> EvidenceRecord:
    score = sum(metrics.values()) / max(len(metrics), 1)
    return EvidenceRecord(technique=technique, status="unavailable", score=score, summary=summary, metrics=metrics)


def _fallback_score(config: dict[str, Any]) -> float:
    fallback = config.get("fallback", {})
    return float(fallback.get("missing_backend_default", 0.30))


def _adapter_path(model_config: dict[str, Any]) -> Path | None:
    value = model_config.get("adapter_path") or model_config.get("lora_path")
    if not value:
        return None
    return Path(str(value))


def _adapter_target_modules(adapter_path: Path) -> list[str]:
    cfg_path = adapter_path / "adapter_config.json"
    if not cfg_path.exists():
        return []
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    targets = payload.get("target_modules", [])
    if isinstance(targets, str):
        return [targets]
    if isinstance(targets, list):
        return [str(item) for item in targets]
    return []

