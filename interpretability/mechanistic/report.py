from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from interpretability.config import load_yaml
from interpretability.mechanistic import analyzers
from interpretability.mechanistic.schema import EvidenceRecord, MechanisticReport, Recommendation

Analyzer = Callable[[Path, dict[str, Any], dict[str, Any]], EvidenceRecord]


def write_mechanistic_report(
    run_dir: str | Path,
    config: dict[str, Any] | str | Path | None,
    model_config: dict[str, Any] | None = None,
) -> MechanisticReport:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    mech_config = _mechanistic_config(config)
    model = dict(model_config or {})
    evidence = _run_analyzers(run_path, mech_config, model)
    report = build_report(run_path, model, evidence)
    report.write(run_path / "mechanistic.json")
    return report


def build_report(run_dir: Path, model_config: dict[str, Any], evidence: list[EvidenceRecord]) -> MechanisticReport:
    scores = _aggregate_scores(evidence)
    legacy = {
        "sparsity": scores.get("feature_sparsity", 0.30),
        "stability": scores.get("attribution_stability", 0.30),
    }
    return MechanisticReport(
        version=1,
        run_dir=str(run_dir),
        model={key: value for key, value in model_config.items() if key not in {"api_key"}},
        evidence=evidence,
        recommendations=_recommendations(evidence, scores),
        scores=scores,
        legacy=legacy,
    )


def _run_analyzers(run_dir: Path, config: dict[str, Any], model_config: dict[str, Any]) -> list[EvidenceRecord]:
    selected = set(config.get("enabled_analyzers", []))
    all_enabled = not selected
    analyzer_map: dict[str, Analyzer] = {
        "lora_delta_geometry": lambda path, cfg, model: analyzers.lora_delta_geometry(path, cfg, model),
        "activation_sparse_autoencoder": lambda path, cfg, model: analyzers.activation_sae(cfg),
        "circuit_tracing": lambda path, cfg, model: analyzers.circuit_tracing(cfg),
        "transcoder_candidates": lambda path, cfg, model: analyzers.transcoder_candidates(cfg),
        "sae_assisted_circuit_discovery": lambda path, cfg, model: analyzers.sae_assisted_circuits(cfg),
        "weight_sparse_base_model_plan": lambda path, cfg, model: analyzers.sparse_training_plan(path, cfg),
        "pruning_masks": lambda path, cfg, model: analyzers.pruning_masks(path, cfg),
    }
    return [
        fn(run_dir, config, model_config)
        for name, fn in analyzer_map.items()
        if all_enabled or name in selected
    ]


def _aggregate_scores(evidence: list[EvidenceRecord]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for item in evidence:
        for key, value in item.metrics.items():
            buckets.setdefault(key, []).append(max(0.0, min(1.0, float(value))))
    defaults = {
        "adapter_locality": 0.30,
        "feature_sparsity": 0.30,
        "circuit_compactness": 0.30,
        "attribution_stability": 0.30,
        "pruning_retained_functionality": 0.30,
        "replacement_readiness": 0.30,
    }
    return {key: sum(values) / len(values) if values else defaults[key] for key, values in ((k, buckets.get(k, [])) for k in defaults)}


def _recommendations(evidence: list[EvidenceRecord], scores: dict[str, float]) -> list[Recommendation]:
    artifact_by_kind = {kind: artifact for item in evidence for kind, artifact in item.artifacts.items()}
    return [
        Recommendation(
            kind="pruning_mask",
            readiness=scores["pruning_retained_functionality"],
            summary="Validate exported masks on task prompts before applying them to the base model.",
            artifact=artifact_by_kind.get("pruning_mask_spec"),
        ),
        Recommendation(
            kind="transcoder_replacement",
            readiness=scores["replacement_readiness"],
            summary="Collect MLP activations before training replacement transcoders.",
            artifact=artifact_by_kind.get("sparse_training_config"),
        ),
    ]


def _mechanistic_config(config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None:
        path = Path("configs/mechanistic.yaml")
        return load_yaml(path) if path.exists() else {}
    if isinstance(config, (str, Path)):
        return load_yaml(config)
    if "mechanistic" in config and isinstance(config["mechanistic"], dict):
        return dict(config["mechanistic"])
    return dict(config)
