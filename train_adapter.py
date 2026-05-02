from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from interpretability.config import load_yaml


def _select_training_runtime(torch_module: Any) -> dict[str, Any]:
    if torch_module.cuda.is_available():
        return {"model_kwargs": {"device_map": "auto"}, "bf16": False, "fp16": True, "use_cpu": False}
    return {"model_kwargs": {}, "bf16": False, "fp16": False, "use_cpu": True}


def _load_examples(path: Path) -> list[dict[str, str]]:
    return [{"text": example["text"]} for example, _line in _load_examples_with_lines(path)]


def _load_examples_with_lines(path: Path) -> list[tuple[dict[str, str], int]]:
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}. Add JSONL training examples before training.")
    examples: list[tuple[dict[str, str], int]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc.msg}") from exc
            if not isinstance(item, dict):
                raise SystemExit(f"Expected object at {path}:{line_no}")
            prompt = item.get("prompt") or item.get("instruction")
            response = item.get("response") or item.get("completion")
            if not isinstance(prompt, str) or not isinstance(response, str):
                raise SystemExit(
                    f"Expected prompt/instruction and response/completion strings at {path}:{line_no}"
                )
            examples.append(({"text": _format_example(prompt, response), "prompt": prompt, "response": response}, line_no))
    if not examples:
        raise SystemExit(f"No training examples found in {path}")
    return examples


def _format_example(prompt: str, response: str) -> str:
    return (
        "<|im_start|>user\n"
        f"{prompt.strip()}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{response.strip()}\n"
        "<|im_end|>"
    )


def _validate_config(config: dict[str, Any]) -> None:
    required = ["model_name", "output_dir", "dataset_path"]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise SystemExit(f"Adapter config missing required keys: {', '.join(missing)}")
    lora = config.get("lora")
    if not isinstance(lora, dict):
        raise SystemExit("Adapter config must include a lora mapping")
    target_modules = lora.get("target_modules")
    if not isinstance(target_modules, list) or not all(isinstance(name, str) for name in target_modules):
        raise SystemExit("Adapter config lora.target_modules must be a list of strings")


def _preflight(config: dict[str, Any]) -> dict[str, Any]:
    dataset_path = Path(config["dataset_path"])
    examples_with_lines = _load_examples_with_lines(dataset_path)
    examples = [example for example, _line in examples_with_lines]
    warnings: list[str] = []
    if len(examples) < 5:
        warnings.append(f"tiny dataset: {len(examples)} example(s)")
    duplicate_lines = _duplicate_lines(examples_with_lines)
    if duplicate_lines:
        warnings.append(f"duplicate rows: {duplicate_lines}")
    if not all(_prompt_matches_harness_shape(example["prompt"]) for example in examples):
        warnings.append("prompt-shape mismatch: expected real harness workspace snapshot prompts")
    guardrail_path = config.get("guardrail_dataset_path")
    guardrail_exists = bool(isinstance(guardrail_path, str) and Path(guardrail_path).exists())
    if not guardrail_exists:
        warnings.append("held-out guardrail set missing")
    token_counts = [len(example["text"].split()) for example in examples]
    metadata = {
        "base_model": config["model_name"],
        "dataset_path": str(dataset_path),
        "dataset_sha256": _file_sha256(dataset_path),
        "dataset_size": len(examples),
        "token_count_words": {
            "min": min(token_counts),
            "max": max(token_counts),
            "avg": sum(token_counts) / len(token_counts),
        },
        "lora": config["lora"],
        "max_steps": int(config.get("max_steps", 20)),
        "output_dir": str(config["output_dir"]),
        "guardrail_dataset_path": str(guardrail_path) if guardrail_path else None,
        "guardrail_dataset_exists": guardrail_exists,
        "warnings": warnings,
    }
    return metadata


def _duplicate_lines(examples_with_lines: list[tuple[dict[str, str], int]]) -> list[list[int]]:
    seen: dict[str, int] = {}
    duplicates: list[list[int]] = []
    for example, line_no in examples_with_lines:
        digest = hashlib.sha256(example["text"].encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates.append([seen[digest], line_no])
        else:
            seen[digest] = line_no
    return duplicates


def _prompt_matches_harness_shape(prompt: str) -> bool:
    required = ["Task instruction:", "Working directory:", "Workspace files:", "Produce the next complete action as JSON"]
    return all(term in prompt for term in required)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adapter.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    _validate_config(config)
    preflight = _preflight(config)
    examples = _load_examples(Path(config["dataset_path"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        for warning in preflight["warnings"]:
            print(f"Warning: {warning}")
        print(
            f"Adapter dry run for {config['model_name']} -> {config['output_dir']} "
            f"with {len(examples)} examples"
        )
        return
    try:
        from datasets import Dataset
        from peft import LoraConfig
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Adapter training requires optional dependencies from requirements.txt. "
            "Run: pip install -r requirements.txt"
        ) from exc
    runtime = _select_training_runtime(torch)
    if runtime["use_cpu"]:
        print("Warning: PyTorch cannot use CUDA; falling back to CPU adapter training.")
    model_name = str(config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        **runtime["model_kwargs"],
    )
    lora = config["lora"]
    peft_config = LoraConfig(
        r=int(lora.get("r", 8)),
        lora_alpha=int(lora.get("alpha", 16)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_args = SFTConfig(
        output_dir=str(config["output_dir"]),
        max_steps=int(config.get("max_steps", 20)),
        learning_rate=float(config.get("learning_rate", 0.0002)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        logging_steps=int(config.get("logging_steps", 1)),
        save_steps=int(config.get("save_steps", 20)),
        report_to="none",
        dataset_text_field="text",
        max_length=int(config.get("max_seq_length", 1024)),
        bf16=runtime["bf16"],
        fp16=runtime["fp16"],
        use_cpu=runtime["use_cpu"],
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(examples),
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(str(config["output_dir"]))
    tokenizer.save_pretrained(str(config["output_dir"]))
    history = getattr(trainer.state, "log_history", [])
    metadata = dict(preflight)
    metadata["loss_history"] = history
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Saved LoRA adapter to {config['output_dir']}")


if __name__ == "__main__":
    main()
