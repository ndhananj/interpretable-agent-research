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

Install vLLM into the same active `.venv` because it is CUDA and platform
sensitive. Do not treat `requirements-vllm.txt` as a second environment recipe;
it is only a deprecated optional marker. Let `uv` choose a vLLM/PyTorch stack
that matches the installed NVIDIA driver:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uv pip install vllm --torch-backend=auto
python serve_vllm_adapter.py --config configs/vllm.yaml --dry-run
python serve_vllm_adapter.py --config configs/vllm.yaml
```

The default vLLM config sets `model.dtype: half` and conservative low-VRAM
limits for a 6 GiB RTX 2060 class GPU, so the dry run includes:

```bash
vllm serve Qwen/Qwen2.5-Coder-1.5B-Instruct --enable-lora --lora-modules interpretable-agent-lora=.../adapters/latest --host 127.0.0.1 --port 8000 --dtype half --max-model-len 1024 --max-num-batched-tokens 1024 --max-num-seqs 1 --gpu-memory-utilization 0.8 --enforce-eager
```

RTX 20xx/Turing GPUs such as the RTX 2060 do not support the model's automatic
`bfloat16` serve path. Keep `model.dtype: half` in `configs/vllm.yaml`, or pass
`--dtype half` when launching the helper.

The same config also reduces Qwen's default context and batching. On 6 GiB
GPUs, Qwen's 32k context defaults can fail during vLLM startup profiling before
the OpenAI server is reachable. This is a serve-time memory setting issue, not
an adapter training issue, so `adapters/latest` does not need to be retrained
for that failure.

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

### Compare base vs LoRA automatically

Use the comparison helper to start the configured vLLM server, verify that both
the base model and LoRA adapter are exposed by `/v1/models`, run the configured
task suite against each model, and write side-by-side reports:

```bash
python compare_vllm_models.py --config configs/vllm.yaml --run-dir runs/model-compare
```

The helper uses `configs/vllm.yaml` as the source of truth. For the base-model
run it removes `model.adapter_name` and `model.adapter_path` in memory, so the
repository config is not changed. For the LoRA run it uses the config as-is and
sends `model.adapter_name` as the OpenAI-compatible model id.

If vLLM is already running, skip process startup and only run readiness/model
checks plus evaluation:

```bash
python compare_vllm_models.py \
  --config configs/vllm.yaml \
  --run-dir runs/model-compare \
  --skip-server-start
```

Outputs are written under the selected run directory:

- `base/<task-name>/` and `lora/<task-name>/`: per-task traces, tool logs,
  metrics, and score files.
- `summary.json`: machine-readable aggregate and per-task functionality,
  explainability, scoring details, acceptance reasons, and run directories.
- `report.md`: compact base-vs-LoRA table for quick inspection.

Functionality is the task check score. Explainability is currently behavioral:
it scores the decision trace and tool log alignment, plus configured default
mechanistic values unless a task run provides `mechanistic.json`. When
mechanistic instrumentation is added, those `mechanistic.json` values will feed
the same scoring path.

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

3. Install vLLM in the same active `.venv` and serve the adapter:

Default keep-driver path:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uv pip install vllm --torch-backend=auto
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python serve_vllm_adapter.py --config configs/vllm.yaml --dry-run
python serve_vllm_adapter.py --config configs/vllm.yaml
```

If this environment already contains the CUDA 13 stack, remove it first:

```bash
pip uninstall -y vllm torch torchvision torchaudio cuda-toolkit cuda-bindings cuda-python
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
when needed. Override dtype the same way if you need a different vLLM dtype:

```bash
python serve_vllm_adapter.py --config configs/vllm.yaml --host 0.0.0.0 --port 8000
python serve_vllm_adapter.py --config configs/vllm.yaml --dtype half
```

For small GPUs, keep `model.max_model_len`, `model.max_num_batched_tokens`,
`model.max_num_seqs`, `model.gpu_memory_utilization`, and
`model.enforce_eager` conservative until vLLM starts reliably. Raise
`max_model_len` and `max_num_batched_tokens` together later only if the workload
needs more context and startup still fits in VRAM. Keep `model.max_tokens` below
`model.max_model_len` so requests leave room for the prompt.

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
- `maximum context length ... requested ... messages ... completion`: vLLM
  requires `prompt_tokens + max_tokens <= max_model_len`. Lower
  `model.max_tokens`, or raise both `model.max_model_len` and
  `model.max_num_batched_tokens` if VRAM allows.
- `vLLM CUDA preflight failed`: the helper found a PyTorch/CUDA/driver mismatch
  before starting vLLM. The known bad local combination is `torch 2.11.0+cu130`
  / CUDA 13 packages on a CUDA 12.2-era driver. Reinstall a driver-compatible
  vLLM/PyTorch stack, or update the NVIDIA driver. Use `--skip-preflight` only
  when you need the raw vLLM startup error for debugging.
- `Bfloat16 is only supported on GPUs with compute capability of at least 8.0`:
  keep `model.dtype: half` in `configs/vllm.yaml`, or launch with
  `--dtype half` on RTX 20xx/Turing GPUs.
- vLLM fails during startup profiling or reports insufficient KV cache memory:
  keep the low-VRAM defaults in `configs/vllm.yaml`, especially
  `model.max_model_len: 1024`, `model.max_num_batched_tokens: 1024`,
  `model.max_num_seqs: 1`, `model.gpu_memory_utilization: 0.80`, and
  `model.enforce_eager: true`. This does not require retraining
  `adapters/latest`. If you need a longer context later, raise
  `model.max_num_batched_tokens` with `model.max_model_len`; vLLM rejects
  `max_num_batched_tokens` values below `max_model_len`.
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
