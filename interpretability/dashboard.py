from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from interpretability.artifacts import read_json, read_jsonl


STATIC_DIR = Path(__file__).with_name("dashboard_static")


@dataclass(frozen=True)
class RunCandidate:
    path: Path
    kind: str
    accepted: bool
    functionality: float
    explainability: float
    timestamp: str
    model_id: str | None
    comparison_role: str | None = None
    comparison_root: Path | None = None

    @property
    def rank_key(self) -> tuple[int, float, float, str, str]:
        return (
            1 if self.accepted else 0,
            self.functionality,
            self.explainability,
            self.timestamp,
            str(self.path),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "accepted": self.accepted,
            "functionality": self.functionality,
            "explainability": self.explainability,
            "timestamp": self.timestamp,
            "model_id": self.model_id,
            "comparison_role": self.comparison_role,
            "comparison_root": str(self.comparison_root) if self.comparison_root else None,
            "rank_key": list(self.rank_key),
        }


@dataclass(frozen=True)
class RunSelection:
    selected: RunCandidate | None
    candidates: list[RunCandidate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict() if self.selected else None,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def select_latest_best_run(runs_root: Path) -> RunSelection:
    candidates = discover_run_candidates(runs_root)
    selected = max(candidates, key=lambda candidate: candidate.rank_key) if candidates else None
    return RunSelection(selected=selected, candidates=sorted(candidates, key=lambda item: item.rank_key, reverse=True))


def discover_run_candidates(runs_root: Path) -> list[RunCandidate]:
    root = Path(runs_root)
    if not root.exists():
        return []
    score_paths = sorted(root.rglob("score.json"))
    return [_candidate_from_score(path.parent, root) for path in score_paths]


def build_dashboard_payload(run_path: Path, runs_root: Path | None = None) -> dict[str, Any]:
    path = Path(run_path)
    score = _read_json_dict(path / "score.json")
    metrics = _read_json_dict(path / "metrics.json")
    config = _read_json_dict(path / "config.json")
    mechanistic = _read_json_dict(path / "mechanistic.json")
    comparison = _comparison_context(path, runs_root or _infer_runs_root(path))
    candidate = _candidate_from_score(path, runs_root or _infer_runs_root(path))
    adapter_metadata = _adapter_metadata(config, mechanistic, path)

    payload = {
        "run": {
            **candidate.to_dict(),
            "name": path.name,
            "absolute_path": str(path.resolve()),
            "relative_path": _relative_to(path, Path.cwd()),
        },
        "model": _model_identity(config, mechanistic, comparison, adapter_metadata),
        "score": {
            "functionality": _float(score.get("functionality", metrics.get("functionality", 0.0))),
            "explainability": _float(score.get("explainability", 0.0)),
            "accepted": bool(score.get("accepted", False)),
            "reason": str(score.get("reason", "missing score")),
            "details": _dict_of_numbers(score.get("details", {})),
        },
        "metrics": metrics,
        "task_checks": _read_json_list(path / "check_diagnostics.json"),
        "decision_trace": _read_text(path / "decision_trace.md"),
        "parsed_action": _read_json_dict(path / "parsed_action.json"),
        "raw_response": {
            "available": (path / "raw_response.txt").exists(),
            "valid_json_action": (path / "parsed_action.json").exists() if (path / "raw_response.txt").exists() else None,
            "text": _read_text(path / "raw_response.txt"),
        },
        "tool_log": read_jsonl(path / "tool_log.jsonl"),
        "file_snapshots": read_json(path / "final_file_snapshots.json", {}),
        "work_files": _work_files(path),
        "mechanistic": _mechanistic_payload(mechanistic),
        "adapter_metadata": adapter_metadata,
        "comparison": comparison,
        "warnings": _payload_warnings(path, score, mechanistic, comparison),
    }
    return payload


def render_full_report(payload: dict[str, Any]) -> str:
    run = payload["run"]
    score = payload["score"]
    model = payload["model"]
    lines = [
        "# Latest Best Model Report",
        "",
        f"- Run: `{run['path']}`",
        f"- Model: `{model.get('model_id') or 'unknown'}`",
        f"- Accepted: `{score['accepted']}` ({score['reason']})",
        f"- Functionality: `{score['functionality']:.3f}`",
        f"- Explainability: `{score['explainability']:.3f}`",
        "",
        "## Score Breakdown",
        "",
    ]
    if score["details"]:
        for key, value in sorted(score["details"].items()):
            lines.append(f"- `{key}`: `{value:.3f}`")
    else:
        lines.append("- No score component details were captured.")
    lines.extend(["", "## Artifacts", ""])
    for label, name in (
        ("Score", "score.json"),
        ("Metrics", "metrics.json"),
        ("Mechanistic report", "mechanistic.json"),
        ("Decision trace", "decision_trace.md"),
        ("Tool log", "tool_log.jsonl"),
        ("Parsed action", "parsed_action.json"),
        ("Check diagnostics", "check_diagnostics.json"),
        ("File snapshots", "final_file_snapshots.json"),
    ):
        artifact_path = Path(run["path"]) / name
        state = "available" if artifact_path.exists() else "missing"
        lines.append(f"- {label}: `{artifact_path}` ({state})")
    lines.extend(["", "## Mechanistic Evidence", ""])
    evidence = payload["mechanistic"]["evidence"]
    if evidence:
        for item in evidence:
            lines.append(f"- `{item.get('technique', 'unknown')}`: {item.get('status', 'unknown')} - {item.get('summary', '')}")
    else:
        lines.append("- No mechanistic evidence records were captured for this run.")
    if payload.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    return "\n".join(lines) + "\n"


def serve_dashboard(runs_root: Path, host: str, port: int, selected_run: Path | None, report_out: Path | None) -> None:
    if report_out is not None:
        selection = select_latest_best_run(runs_root)
        run = selected_run or (selection.selected.path if selection.selected else None)
        if run is None:
            raise SystemExit(f"No runs with score.json found under {runs_root}")
        report_out.write_text(render_full_report(build_dashboard_payload(run, runs_root)), encoding="utf-8")
    handler = _make_handler(runs_root, selected_run)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving interpretability dashboard at http://{host}:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local dashboard for the latest best model run")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--run", default=None, help="Inspect a specific run directory instead of auto-selecting best.")
    parser.add_argument("--report-out", default=None, help="Write a Markdown report for the selected run.")
    args = parser.parse_args()
    serve_dashboard(
        runs_root=Path(args.runs_root),
        host=args.host,
        port=args.port,
        selected_run=Path(args.run) if args.run else None,
        report_out=Path(args.report_out) if args.report_out else None,
    )


def _make_handler(runs_root: Path, selected_run: Path | None) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._write_bytes((STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif parsed.path == "/dashboard.js":
                self._write_bytes((STATIC_DIR / "dashboard.js").read_bytes(), "text/javascript; charset=utf-8")
            elif parsed.path == "/dashboard.css":
                self._write_bytes((STATIC_DIR / "dashboard.css").read_bytes(), "text/css; charset=utf-8")
            elif parsed.path == "/api/runs":
                self._write_json(select_latest_best_run(runs_root).to_dict())
            elif parsed.path == "/api/best":
                selection = select_latest_best_run(runs_root)
                selected = selected_run or (selection.selected.path if selection.selected else None)
                if selected is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "No runs with score.json found")
                    return
                self._write_json(build_dashboard_payload(selected, runs_root))
            elif parsed.path == "/api/run":
                query = parse_qs(parsed.query)
                values = query.get("path", [])
                if not values:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Missing path query parameter")
                    return
                self._write_json(build_dashboard_payload(_resolve_requested_path(values[0], runs_root), runs_root))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _write_json(self, payload: dict[str, Any]) -> None:
            self._write_bytes(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"), "application/json; charset=utf-8")

        def _write_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DashboardHandler


def _candidate_from_score(run_dir: Path, runs_root: Path) -> RunCandidate:
    score = _read_json_dict(run_dir / "score.json")
    metrics = _read_json_dict(run_dir / "metrics.json")
    config = _read_json_dict(run_dir / "config.json")
    comparison_root, role = _comparison_role(run_dir, runs_root)
    return RunCandidate(
        path=run_dir,
        kind="model_comparison_task" if comparison_root else "trial",
        accepted=bool(score.get("accepted", False)),
        functionality=_float(score.get("functionality", metrics.get("functionality", 0.0))),
        explainability=_float(score.get("explainability", 0.0)),
        timestamp=_timestamp_for(run_dir),
        model_id=_model_identity(config, {}, _comparison_context(run_dir, runs_root), None).get("model_id"),
        comparison_role=role,
        comparison_root=comparison_root,
    )


def _comparison_context(run_dir: Path, runs_root: Path) -> dict[str, Any] | None:
    root, role = _comparison_role(run_dir, runs_root)
    if root is None:
        return None
    summary = _read_json_dict(root / "summary.json")
    return {"root": str(root), "role": role, "summary": summary}


def _comparison_role(run_dir: Path, runs_root: Path) -> tuple[Path | None, str | None]:
    try:
        relative = run_dir.relative_to(runs_root)
    except ValueError:
        relative = run_dir
    parts = relative.parts
    if len(parts) >= 3 and parts[-3].startswith("model-compare") and parts[-2] in {"base", "lora"}:
        return runs_root / Path(*parts[:-2]), parts[-2]
    return None, None


def _model_identity(
    config: dict[str, Any],
    mechanistic: dict[str, Any],
    comparison: dict[str, Any] | None,
    adapter_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    model_cfg = config.get("model", config) if isinstance(config.get("model", config), dict) else {}
    mech_model = mechanistic.get("model", {}) if isinstance(mechanistic.get("model", {}), dict) else {}
    model_id = model_cfg.get("name") or mech_model.get("name")
    if comparison and comparison.get("summary"):
        role = comparison.get("role")
        summary_key = "lora_model" if role == "lora" else "base_model"
        model_id = comparison["summary"].get(summary_key, {}).get("model_id", model_id)
    return {
        "model_id": model_id,
        "backend": model_cfg.get("backend") or mech_model.get("backend"),
        "adapter_name": model_cfg.get("adapter_name") or (adapter_metadata or {}).get("base_model_name_or_path"),
        "adapter_path": model_cfg.get("adapter_path"),
        "config": model_cfg,
    }


def _mechanistic_payload(mechanistic: dict[str, Any]) -> dict[str, Any]:
    evidence = mechanistic.get("evidence", [])
    recommendations = mechanistic.get("recommendations", [])
    return {
        "available": bool(mechanistic),
        "scores": _dict_of_numbers(mechanistic.get("scores", {})),
        "evidence": evidence if isinstance(evidence, list) else [],
        "recommendations": recommendations if isinstance(recommendations, list) else [],
        "deep_circuit_backends": {
            "activation_capture": "unavailable",
            "sae_training": "unavailable",
            "circuit_graph_generation": "unavailable",
        },
    }


def _adapter_metadata(config: dict[str, Any], mechanistic: dict[str, Any], run_dir: Path) -> dict[str, Any] | None:
    model_cfg = config.get("model", config) if isinstance(config.get("model", config), dict) else {}
    candidates = [
        model_cfg.get("adapter_path"),
        mechanistic.get("model", {}).get("adapter_path") if isinstance(mechanistic.get("model"), dict) else None,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        metadata = read_json(path / "adapter_config.json", None)
        if metadata:
            return {"path": str(path), "adapter_config": metadata}
    return None


def _payload_warnings(path: Path, score: dict[str, Any], mechanistic: dict[str, Any], comparison: dict[str, Any] | None) -> list[str]:
    warnings: list[str] = []
    if not score:
        warnings.append("score.json is missing or empty")
    if not mechanistic:
        warnings.append("mechanistic.json is missing; deep evidence panels are empty")
    warnings.append("Activation capture, SAE training, and circuit graph generation are not implemented in this dashboard version")
    if comparison and comparison.get("summary", {}).get("warnings"):
        warnings.extend(str(item) for item in comparison["summary"]["warnings"])
    return warnings


def _work_files(run_dir: Path) -> list[dict[str, Any]]:
    work = run_dir / "work"
    if not work.exists():
        return []
    files = []
    for path in sorted(item for item in work.rglob("*") if item.is_file()):
        text = _read_text(path)
        files.append({"path": str(path.relative_to(work)), "excerpt": text[:2000], "size": path.stat().st_size})
    return files


def _resolve_requested_path(value: str, runs_root: Path) -> Path:
    path = Path(unquote(value))
    if path.is_absolute():
        return path
    return Path.cwd() / path if path.parts and path.parts[0] == str(runs_root) else runs_root / path


def _infer_runs_root(path: Path) -> Path:
    parts = path.parts
    if "runs" in parts:
        return Path(*parts[: parts.index("runs") + 1])
    return path.parent


def _timestamp_for(path: Path) -> str:
    match = re.search(r"(20\d{6}T\d{6})", str(path))
    if match:
        return match.group(1)
    try:
        return f"{path.stat().st_mtime:.6f}"
    except OSError:
        return ""


def _read_json_dict(path: Path) -> dict[str, Any]:
    value = read_json(path, {})
    return value if isinstance(value, dict) else {}


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    value = read_json(path, [])
    return value if isinstance(value, list) else []


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _dict_of_numbers(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _float(item) for key, item in value.items()}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def html_escape(value: Any) -> str:
    return html.escape(str(value))


if __name__ == "__main__":
    main()
