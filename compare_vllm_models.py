from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent_harness.backends import make_backend
from agent_harness.task import run_task
from interpretability.config import (
    experiment_config,
    load_yaml,
    model_config,
    resources_config,
    scoring_config,
)
from interpretability.scoring import ScoreResult, score_run


@dataclass(frozen=True)
class TaskComparison:
    task: str
    base: dict[str, Any]
    lora: dict[str, Any]


@dataclass(frozen=True)
class ModelSummary:
    model_id: str
    run_dir: str
    functionality: float
    explainability: float
    accepted_tasks: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare configured base and LoRA models through vLLM")
    parser.add_argument("--config", default="configs/vllm.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--startup-timeout-s", type=float, default=180.0)
    parser.add_argument("--keep-server-running", action="store_true")
    parser.add_argument("--skip-server-start", action="store_true")
    parser.add_argument("--task", action="append", default=None, help="Task path override; may be repeated.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else Path(f"runs/model-compare-{time.strftime('%Y%m%dT%H%M%S')}")
    summary = compare_models(
        config_path=args.config,
        run_dir=run_dir,
        startup_timeout_s=args.startup_timeout_s,
        keep_server_running=args.keep_server_running,
        skip_server_start=args.skip_server_start,
        task_overrides=args.task,
    )
    print_stdout_summary(summary)


def compare_models(
    *,
    config_path: str | Path,
    run_dir: Path,
    startup_timeout_s: float = 180.0,
    keep_server_running: bool = False,
    skip_server_start: bool = False,
    task_overrides: list[str] | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    model = model_config(config)
    base_model_id = _required_model_id(model, "name")
    lora_model_id = _required_model_id(model, "adapter_name")
    base_url = str(model.get("base_url", "http://127.0.0.1:8000")).rstrip("/")

    scoring = scoring_config(config)
    resources = resources_config(config)
    exp = experiment_config(config)
    metrics_config = load_yaml(scoring.metrics_config)
    task_paths = task_overrides or scoring.task_paths
    run_dir.mkdir(parents=True, exist_ok=True)

    server: subprocess.Popen[str] | None = None
    try:
        if not skip_server_start:
            server = start_server(config_path)
        wait_for_health(base_url, startup_timeout_s)
        available_models = fetch_model_ids(base_url)
        missing = [model_id for model_id in (base_model_id, lora_model_id) if model_id not in available_models]
        if missing:
            raise SystemExit(f"vLLM server is missing required model id(s): {', '.join(missing)}")

        base_config = _base_model_config(model)
        lora_config = dict(model)
        comparisons = [
            run_task_pair(
                task_path=task_path,
                base_config=base_config,
                lora_config=lora_config,
                run_dir=run_dir,
                timeout_s=resources.task_timeout_s,
                metrics_config=metrics_config,
                floor_ratio=exp.functionality_floor_ratio,
            )
            for task_path in task_paths
        ]
        summary = build_summary(
            run_dir=run_dir,
            base_model_id=base_model_id,
            lora_model_id=lora_model_id,
            comparisons=comparisons,
            available_models=available_models,
        )
        write_reports(run_dir, summary)
        return summary
    finally:
        if server is not None and not keep_server_running:
            stop_server(server)


def start_server(config_path: str | Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "serve_vllm_adapter.py", "--config", str(config_path)],
        text=True,
    )


def stop_server(server: subprocess.Popen[str]) -> None:
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=20)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=10)


def wait_for_health(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(1)
    message = f"Timed out waiting for vLLM health at {base_url}/health after {timeout_s:g}s"
    if last_error is not None:
        message = f"{message}: {last_error}"
    raise SystemExit(message)


def fetch_model_ids(base_url: str) -> set[str]:
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read vLLM model list from {base_url}/v1/models: {exc}") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise SystemExit("vLLM /v1/models response did not contain a data list")
    return {str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id") is not None}


def run_task_pair(
    *,
    task_path: str,
    base_config: dict[str, Any],
    lora_config: dict[str, Any],
    run_dir: Path,
    timeout_s: int,
    metrics_config: dict[str, Any],
    floor_ratio: float,
) -> TaskComparison:
    task_name = _task_name(task_path)
    base_dir = run_dir / "base" / task_name
    lora_dir = run_dir / "lora" / task_name
    base_score = run_model_task(task_path, base_config, base_dir, timeout_s, metrics_config, floor_ratio, None)
    lora_score = run_model_task(
        task_path,
        lora_config,
        lora_dir,
        timeout_s,
        metrics_config,
        floor_ratio,
        base_score.functionality,
    )
    return TaskComparison(
        task=task_path,
        base=_score_payload(base_score, base_dir),
        lora=_score_payload(lora_score, lora_dir),
    )


def run_model_task(
    task_path: str,
    model_cfg: dict[str, Any],
    run_dir: Path,
    timeout_s: int,
    metrics_config: dict[str, Any],
    floor_ratio: float,
    baseline_functionality: float | None,
) -> ScoreResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    result = run_task(task_path, make_backend(model_cfg), run_dir, timeout_s)
    (run_dir / "metrics.json").write_text(
        json.dumps({"functionality": result.functionality}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(json.dumps({"model": model_cfg}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if baseline_functionality is None:
        score = score_run(run_dir, metrics_config, result.functionality, floor_ratio, incumbent_explainability=-1.0)
        score = ScoreResult(score.functionality, score.explainability, True, "baseline", score.details)
    else:
        score = score_run(run_dir, metrics_config, baseline_functionality, floor_ratio, incumbent_explainability=-1.0)
    (run_dir / "score.json").write_text(json.dumps(asdict(score), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return score


def build_summary(
    *,
    run_dir: Path,
    base_model_id: str,
    lora_model_id: str,
    comparisons: list[TaskComparison],
    available_models: set[str],
) -> dict[str, Any]:
    task_payloads = [asdict(item) for item in comparisons]
    base_summary = _model_summary(base_model_id, run_dir / "base", [item.base for item in comparisons])
    lora_summary = _model_summary(lora_model_id, run_dir / "lora", [item.lora for item in comparisons])
    return {
        "base_model": asdict(base_summary),
        "lora_model": asdict(lora_summary),
        "available_models": sorted(available_models),
        "tasks": task_payloads,
    }


def write_reports(run_dir: Path, summary: dict[str, Any]) -> None:
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(render_markdown_report(summary), encoding="utf-8")


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Base vs LoRA Comparison",
        "",
        "| Model | Functionality | Explainability | Accepted tasks | Run dir |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key in ("base_model", "lora_model"):
        model = summary[key]
        lines.append(
            f"| {model['model_id']} | {model['functionality']:.3f} | "
            f"{model['explainability']:.3f} | {model['accepted_tasks']} | `{model['run_dir']}` |"
        )
    lines.extend(["", "## Per Task", ""])
    lines.append("| Task | Base functionality | LoRA functionality | Base explainability | LoRA explainability | LoRA reason |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for task in summary["tasks"]:
        lines.append(
            f"| {task['task']} | {task['base']['functionality']:.3f} | {task['lora']['functionality']:.3f} | "
            f"{task['base']['explainability']:.3f} | {task['lora']['explainability']:.3f} | {task['lora']['reason']} |"
        )
    return "\n".join(lines) + "\n"


def print_stdout_summary(summary: dict[str, Any]) -> None:
    base = summary["base_model"]
    lora = summary["lora_model"]
    print(
        "Base vs LoRA: "
        f"{base['model_id']} functionality={base['functionality']:.3f}, explainability={base['explainability']:.3f}; "
        f"{lora['model_id']} functionality={lora['functionality']:.3f}, explainability={lora['explainability']:.3f}"
    )
    print(f"Reports: {Path(base['run_dir']).parent / 'summary.json'} and {Path(base['run_dir']).parent / 'report.md'}")


def _base_model_config(model: dict[str, Any]) -> dict[str, Any]:
    values = dict(model)
    values.pop("adapter_name", None)
    values.pop("adapter_path", None)
    return values


def _required_model_id(model: dict[str, Any], key: str) -> str:
    value = model.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"model.{key} must be configured")
    return value


def _task_name(task_path: str) -> str:
    task = load_yaml(task_path)
    name = task.get("name")
    if isinstance(name, str) and name.strip():
        return name
    return Path(task_path).stem


def _score_payload(score: ScoreResult, run_dir: Path) -> dict[str, Any]:
    return {
        "functionality": score.functionality,
        "explainability": score.explainability,
        "accepted": score.accepted,
        "reason": score.reason,
        "details": score.details,
        "run_dir": str(run_dir),
    }


def _model_summary(model_id: str, run_dir: Path, scores: list[dict[str, Any]]) -> ModelSummary:
    count = max(len(scores), 1)
    return ModelSummary(
        model_id=model_id,
        run_dir=str(run_dir),
        functionality=sum(float(score["functionality"]) for score in scores) / count,
        explainability=sum(float(score["explainability"]) for score in scores) / count,
        accepted_tasks=sum(1 for score in scores if score["accepted"]),
    )


if __name__ == "__main__":
    main()
