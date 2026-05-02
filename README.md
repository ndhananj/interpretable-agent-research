# Interpretable Agent Research

An Autoresearch-style loop for improving a coding and shell-use LLM while making
interpretability the objective and preserving baseline functionality.

This repository uses Andrej Karpathy's `autoresearch` as a reference for the
outer loop pattern: propose one candidate, run a bounded experiment, score it,
keep improvements, and log everything. The target here is different: a small
coding agent with shell/file access, adapter-friendly model configuration, and
explainability metrics.

## Quick Start

### Mock CPU-safe run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py --config configs/default.yaml --max-trials 1
```

The default configuration uses a deterministic mock model, so it runs without
network access or GPU access. Adapter training and vLLM acceleration are
optional paths enabled by configuration and local hardware availability.

### Train a LoRA adapter

`configs/adapter.yaml` defaults to `Qwen/Qwen2.5-Coder-1.5B-Instruct`, reads
JSONL examples from `data/adapter_examples.jsonl`, and writes the PEFT adapter
artifact to `adapters/latest`.

```bash
python train_adapter.py --config configs/adapter.yaml --dry-run
python train_adapter.py --config configs/adapter.yaml
```

Each JSONL row must contain `prompt` or `instruction`, plus `response` or
`completion`. The dry run validates the config and dataset without loading model
weights.

### Serve through local vLLM

Install vLLM separately because it is CUDA and platform sensitive. On this
machine, do not treat `requirements-vllm.txt` as the preferred GPU install path;
it is only a minimal optional marker. Prefer a fresh environment and let `uv`
choose a vLLM/PyTorch stack that matches the installed NVIDIA driver:

```bash
python3 -m venv .venv-vllm
. .venv-vllm/bin/activate
pip install -r requirements.txt
uv pip install vllm --torch-backend=auto
python serve_vllm_adapter.py --config configs/vllm.yaml --dry-run
python serve_vllm_adapter.py --config configs/vllm.yaml
```

If preflight reports `vllm==0.6.6.post1` with an incompatible
`transformers` version, keep `adapters/latest` as-is and only repair the active
vLLM environment:

```bash
uv pip install 'transformers>=4.56.2,<5'
python serve_vllm_adapter.py --config configs/vllm.yaml
```

The current bad state observed on this machine is `torch 2.11.0+cu130` with
CUDA 13 packages on a CUDA 12.2-era NVIDIA driver. Keeping the driver means
reinstalling a driver-compatible vLLM/PyTorch wheel set. The other valid fix is
updating the NVIDIA driver so it supports the installed CUDA 13 runtime, then
rebooting or reloading the driver before verification.

In another shell, point the harness at the OpenAI-compatible localhost server:

```bash
python agent_harness/run_task.py \
  --task tasks/replace_token.yaml \
  --config configs/vllm.yaml \
  --run-dir runs/vllm-task
```

## Continuous agent with adapted vLLM model

This path runs CSI against the LoRA adapter produced by `train_adapter.py`.
`configs/vllm.yaml` keeps the base model in `model.name`, serves the adapter
from `model.adapter_path`, and sends `model.adapter_name` as the OpenAI model
name because vLLM exposes LoRA modules by that name.

1. Set up the base environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

2. Train or refresh the adapter:

```bash
python train_adapter.py --config configs/adapter.yaml --dry-run
python train_adapter.py --config configs/adapter.yaml
```

The default adapter output is `adapters/latest`, matching
`configs/vllm.yaml`.

3. Install vLLM and serve the adapter:

Default keep-driver path:

```bash
python3 -m venv .venv-vllm
. .venv-vllm/bin/activate
pip install -r requirements.txt
uv pip install vllm --torch-backend=auto
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python serve_vllm_adapter.py --config configs/vllm.yaml --dry-run
python serve_vllm_adapter.py --config configs/vllm.yaml
```

If this environment already contains the CUDA 13 stack, remove it first:

```bash
pip uninstall -y vllm torch torchvision torchaudio 'nvidia-*'
uv pip install vllm --torch-backend=auto
```

If preflight instead reports only a package compatibility issue such as
`vllm==0.6.6.post1` with `transformers>=5`, do not retrain the adapter. Repair
the active vLLM environment and rerun serve:

```bash
uv pip install 'transformers>=4.56.2,<5'
python serve_vllm_adapter.py --config configs/vllm.yaml
```

`uv pip install vllm --torch-backend=auto` follows vLLM's GPU install guidance
by selecting a PyTorch backend from the installed driver. If `uv` is not
available, use the official vLLM GPU install docs and PyTorch previous-version
wheel indexes to choose a CUDA runtime compatible with this driver, such as a
CUDA 12.x wheel set instead of CUDA 13.

On the observed driver version `12020`, latest vLLM may still resolve to CUDA
12.8, which is too new for this driver. The verified keep-driver fallback is:

```bash
pip uninstall -y vllm torch torchvision torchaudio cuda-toolkit cuda-bindings cuda-python
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install vllm==0.6.6.post1 --torch-backend=cu121
```

Alternate driver-update path: upgrade the NVIDIA driver to one compatible with
the currently installed CUDA 13 runtime, reboot or reload the driver, then rerun
the verification commands above in the existing or rebuilt vLLM environment. If
`nvidia-smi` cannot communicate with the driver, the setup remains blocked at
the system-driver layer even if Python packages are corrected.

The helper reads `model.base_url` for the default host and port. Override them
when needed:

```bash
python serve_vllm_adapter.py --config configs/vllm.yaml --host 0.0.0.0 --port 8000
```

4. Validate one task before starting the continuous loop:

```bash
python agent_harness/run_task.py \
  --task tasks/replace_token.yaml \
  --config configs/vllm.yaml \
  --run-dir runs/vllm-smoke
```

5. Run one foreground CSI iteration for a smoke test, or start continuous mode:

```bash
python csi.py --config configs/vllm.yaml _daemon --foreground
python csi.py --config configs/vllm.yaml start
python csi.py --config configs/vllm.yaml status
python csi.py --config configs/vllm.yaml dashboard
python csi.py --config configs/vllm.yaml stop
```

The dashboard is served at `http://127.0.0.1:8765` by default.

Common failure modes:

- `model.adapter_path does not exist`: train the adapter first, or update
  `configs/vllm.yaml` to the adapter directory you want to serve.
- `Could not reach local vLLM OpenAI server`: start `serve_vllm_adapter.py`, or
  make `model.base_url` match the vLLM host and port.
- `The model ... does not exist`: confirm `model.adapter_name` matches the
  module name in the helper dry-run command.
- `vLLM CUDA preflight failed`: the helper found a PyTorch/CUDA/driver mismatch
  before starting vLLM. The known bad local combination is `torch 2.11.0+cu130`
  / CUDA 13 packages on a CUDA 12.2-era driver. Reinstall a driver-compatible
  vLLM/PyTorch stack, or update the NVIDIA driver. Use `--skip-preflight` only
  when you need the raw vLLM startup error for debugging.
- CUDA or package import failures: install the optional vLLM stack in the active
  environment and run on a CUDA-capable machine supported by vLLM.

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
python csi.py start
python csi.py status
python csi.py dashboard
python csi.py stop
python agent_harness/run_task.py --task tasks/replace_token.yaml --config configs/default.yaml
python score_candidate.py --run runs/<run-id>
python train_adapter.py --config configs/adapter.yaml --dry-run
python train_adapter.py --config configs/adapter.yaml
python serve_vllm_adapter.py --config configs/vllm.yaml --dry-run
```

The `csi.py` commands run the experiment loop continuously in a local
background process and write control state to `runs/csi/`. The dashboard serves
a standard-library localhost view at `http://127.0.0.1:8765`.

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
