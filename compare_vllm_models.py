from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
            ensure_startup_port_available(base_url)
            server = start_server(config_path)
        wait_for_health(base_url, startup_timeout_s, server)
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


def ensure_startup_port_available(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise SystemExit("model.base_url must be an http(s) URL with a host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=1):
            pass
    except OSError:
        return
    raise SystemExit(
        f"Cannot start vLLM because {parsed.hostname}:{port} is already in use. "
        "Stop the existing server or rerun with --skip-server-start to use it."
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


def wait_for_health(base_url: str, timeout_s: float, server: subprocess.Popen[str] | None = None) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if server is not None and server.poll() is not None:
            raise SystemExit(f"vLLM server exited before becoming healthy with code {server.returncode}")
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
    base_payload = _score_payload(base_score, base_dir)
    lora_payload = _score_payload(lora_score, lora_dir)
    return TaskComparison(
        task=task_path,
        base=base_payload,
        lora=lora_payload,
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
    task_payloads = []
    for item in comparisons:
        payload = asdict(item)
        payload["diagnostics"] = _comparison_diagnostics(payload["base"], payload["lora"])
        task_payloads.append(payload)
    base_summary = _model_summary(base_model_id, run_dir / "base", [item.base for item in comparisons])
    lora_summary = _model_summary(lora_model_id, run_dir / "lora", [item.lora for item in comparisons])
    return {
        "base_model": asdict(base_summary),
        "lora_model": asdict(lora_summary),
        "available_models": sorted(available_models),
        "tasks": task_payloads,
        "warnings": _regression_warnings(task_payloads),
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
    lines.append("| Task | Base functionality | LoRA functionality | Check deltas | Raw JSON | Edit/command diff | Likely failure | LoRA reason |")
    lines.append("| --- | ---: | ---: | --- | --- | --- | --- | --- |")
    for task in summary["tasks"]:
        diagnostics = task.get("diagnostics", {})
        lines.append(
            f"| {task['task']} | {task['base']['functionality']:.3f} | {task['lora']['functionality']:.3f} | "
            f"{_markdown_join(diagnostics.get('check_deltas', []))} | "
            f"{diagnostics.get('raw_response_validity', 'unknown')} | "
            f"{_markdown_join(diagnostics.get('action_differences', []))} | "
            f"{diagnostics.get('likely_failure_category', 'unknown')} | {task['lora']['reason']} |"
        )
    if summary.get("warnings"):
        lines.extend(["", "## Regression Warnings", ""])
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
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
        "check_diagnostics": _read_json(run_dir / "check_diagnostics.json", []),
        "file_snapshots": _read_json(run_dir / "final_file_snapshots.json", {}),
        "raw_response_valid": _raw_response_valid(run_dir),
        "parsed_action": _read_json(run_dir / "parsed_action.json", {}),
        "run_dir": str(run_dir),
    }


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_response_valid(run_dir: Path) -> bool | None:
    if not (run_dir / "raw_response.txt").exists():
        return None
    return (run_dir / "parsed_action.json").exists()


def _comparison_diagnostics(base: dict[str, Any], lora: dict[str, Any]) -> dict[str, Any]:
    check_deltas = _check_deltas(base.get("check_diagnostics", []), lora.get("check_diagnostics", []))
    action_differences = _action_differences(base.get("parsed_action", {}), lora.get("parsed_action", {}))
    return {
        "check_deltas": check_deltas,
        "raw_response_validity": _raw_validity_label(base.get("raw_response_valid"), lora.get("raw_response_valid")),
        "action_differences": action_differences,
        "likely_failure_category": _likely_failure_category(base, lora, check_deltas, action_differences),
    }


def _check_deltas(base_checks: list[dict[str, Any]], lora_checks: list[dict[str, Any]]) -> list[str]:
    base_by_key = {_check_key(check): check for check in base_checks}
    lora_by_key = {_check_key(check): check for check in lora_checks}
    deltas: list[str] = []
    for key in sorted(set(base_by_key) | set(lora_by_key)):
        base_passed = base_by_key.get(key, {}).get("passed")
        lora_passed = lora_by_key.get(key, {}).get("passed")
        if base_passed != lora_passed:
            deltas.append(f"{key}: base={base_passed} lora={lora_passed}")
    return deltas or ["no pass deltas"]


def _check_key(check: dict[str, Any]) -> str:
    return f"{check.get('type')} {check.get('path')} {check.get('expected_text')!r}"


def _action_differences(base_action: dict[str, Any], lora_action: dict[str, Any]) -> list[str]:
    differences: list[str] = []
    if base_action.get("edits") != lora_action.get("edits"):
        differences.append("edits differ")
    if base_action.get("commands") != lora_action.get("commands"):
        differences.append("commands differ")
    if bool(base_action) != bool(lora_action):
        differences.append("parsed action availability differs")
    return differences or ["same edits and commands"]


def _raw_validity_label(base_valid: bool | None, lora_valid: bool | None) -> str:
    return f"base={_validity_value(base_valid)}, lora={_validity_value(lora_valid)}"


def _validity_value(value: bool | None) -> str:
    if value is None:
        return "not captured"
    return "valid" if value else "invalid"


def _likely_failure_category(
    base: dict[str, Any],
    lora: dict[str, Any],
    check_deltas: list[str],
    action_differences: list[str],
) -> str:
    if lora.get("raw_response_valid") is False:
        return "json_format"
    if float(lora.get("functionality", 0.0)) < float(base.get("functionality", 0.0)):
        if action_differences != ["same edits and commands"]:
            return "action_regression"
        if check_deltas != ["no pass deltas"]:
            return "check_regression"
        return "score_regression"
    if float(lora.get("functionality", 0.0)) == 0.0:
        return "task_failed"
    return "none"


def _regression_warnings(tasks: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for task in tasks:
        base = task["base"]
        lora = task["lora"]
        if float(lora["functionality"]) < float(base["functionality"]):
            warnings.append(
                f"{task['task']}: LoRA functionality {lora['functionality']:.3f} regressed below "
                f"base {base['functionality']:.3f}"
            )
        for delta in task.get("diagnostics", {}).get("check_deltas", []):
            if "base=True lora=False" in delta:
                warnings.append(f"{task['task']}: per-check regression {delta}")
    return warnings


def _markdown_join(items: list[str]) -> str:
    return "<br>".join(str(item).replace("|", "\\|") for item in items)


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
