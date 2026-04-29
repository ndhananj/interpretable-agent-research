# Interpretable Agent Research

An Autoresearch-style loop for improving a coding and shell-use LLM while making
interpretability the objective and preserving baseline functionality.

This repository uses Andrej Karpathy's `autoresearch` as a reference for the
outer loop pattern: propose one candidate, run a bounded experiment, score it,
keep improvements, and log everything. The target here is different: a small
coding agent with shell/file access, adapter-friendly model configuration, and
explainability metrics.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py --config configs/default.yaml --max-trials 1
```

The default configuration uses a deterministic mock model, so it runs without
network access or GPU access. Adapter training and vLLM acceleration are
optional paths enabled by configuration and local hardware availability.

## Hardware Policy

The runtime is conservative by default:

- Detects CUDA, `nvidia-smi`, vLLM, RAM, CPU count, and current load.
- Falls back to CPU-safe execution when CUDA is unavailable.
- Caps threads and wall-clock time per task.
- Rejects experiments while memory or load pressure is above configured limits.
- Uses vLLM only for inference when CUDA is actually accessible.

## Objective

Candidates must first pass a baseline-relative functionality gate. The default
floor is:

```text
candidate_functionality >= 0.90 * baseline_functionality
```

Only candidates passing that floor compete on explainability. Explainability is
scored from behavior-level evidence now and mechanistic evidence when the model
backend exposes activations.

## Main Commands

```bash
python run_experiment.py --config configs/default.yaml
python agent_harness/run_task.py --task tasks/replace_token.yaml --config configs/default.yaml
python score_candidate.py --run runs/<run-id>
python train_adapter.py --config configs/adapter.yaml
```

## Layout

- `program.md`: research instructions for candidate generation.
- `configs/`: loop, adapter, and metric configuration.
- `agent_harness/`: shell/file task execution and model backends.
- `interpretability/`: scoring and resource governance.
- `tasks/`: small coding-agent fixtures.
- `tests/`: unit and smoke tests.

## Acknowledgements

This project is inspired by Andrej Karpathy's
[`autoresearch`](https://github.com/karpathy/autoresearch), which provides the
precursor autonomous research loop pattern. This repository adapts that pattern
for interpretable coding-agent research rather than single-GPU nanochat
training.
