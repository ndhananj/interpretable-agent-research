from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from interpretability.artifacts import write_json, write_jsonl
from interpretability.dashboard import (
    _make_handler,
    build_dashboard_payload,
    render_full_report,
    select_latest_best_run,
)


def _write_run(
    run_dir: Path,
    *,
    accepted: bool,
    functionality: float,
    explainability: float,
    reason: str = "accepted",
    model: str = "model",
) -> None:
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "score.json",
        {
            "accepted": accepted,
            "functionality": functionality,
            "explainability": explainability,
            "reason": reason,
            "details": {"behavior_trace_concision": explainability},
        },
        sort_keys=False,
    )
    write_json(run_dir / "metrics.json", {"functionality": functionality}, sort_keys=False)
    write_json(run_dir / "config.json", {"model": {"name": model, "backend": "fake"}}, sort_keys=False)


def test_latest_best_selection_prefers_accepted_then_functionality_then_explainability_then_newest(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_run(root / "20260501T000000-trial-a", accepted=False, functionality=1.0, explainability=1.0)
    _write_run(root / "20260501T000001-trial-b", accepted=True, functionality=0.5, explainability=0.9)
    _write_run(root / "20260501T000002-trial-c", accepted=True, functionality=0.8, explainability=0.1)
    _write_run(root / "20260501T000003-trial-d", accepted=True, functionality=0.8, explainability=0.3)
    _write_run(root / "20260501T000004-trial-e", accepted=True, functionality=0.8, explainability=0.3)

    selection = select_latest_best_run(root)

    assert selection.selected is not None
    assert selection.selected.path.name == "20260501T000004-trial-e"
    assert selection.candidates[0].path == selection.selected.path


def test_model_compare_candidates_are_parsed_with_model_context(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    compare = root / "model-compare"
    _write_run(compare / "base" / "task_a", accepted=True, functionality=0.9, explainability=0.2, model="base-id")
    _write_run(compare / "lora" / "task_a", accepted=True, functionality=0.7, explainability=0.9, model="lora-id")
    write_json(
        compare / "summary.json",
        {
            "base_model": {"model_id": "base-id", "functionality": 0.9, "explainability": 0.2, "accepted_tasks": 1, "run_dir": str(compare / "base")},
            "lora_model": {"model_id": "lora-id", "functionality": 0.7, "explainability": 0.9, "accepted_tasks": 1, "run_dir": str(compare / "lora")},
            "tasks": [],
            "warnings": ["lora regressed"],
        },
        sort_keys=False,
    )

    selection = select_latest_best_run(root)
    payload = build_dashboard_payload(compare / "base" / "task_a", root)

    assert selection.selected is not None
    assert selection.selected.path == compare / "base" / "task_a"
    assert payload["run"]["kind"] == "model_comparison_task"
    assert payload["comparison"]["role"] == "base"
    assert payload["model"]["model_id"] == "base-id"
    assert "lora regressed" in payload["warnings"]


def test_payload_generation_collects_deep_artifacts_and_empty_states(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "20260501T010101-trial-1"
    _write_run(run_dir, accepted=True, functionality=1.0, explainability=0.4)
    write_json(run_dir / "check_diagnostics.json", [{"type": "file_contains", "path": "input.txt", "passed": True}], sort_keys=False)
    write_json(run_dir / "parsed_action.json", {"edits": {"input.txt": "DONE"}, "commands": []}, sort_keys=False)
    write_json(run_dir / "final_file_snapshots.json", {"input.txt": {"exists": True}}, sort_keys=False)
    write_json(
        run_dir / "mechanistic.json",
        {
            "scores": {"feature_sparsity": 0.7},
            "evidence": [{"technique": "lora_delta_geometry", "status": "available", "score": 0.7, "summary": "ok"}],
            "recommendations": [{"kind": "pruning_mask", "readiness": 0.3, "summary": "validate", "artifact": "mask.json"}],
        },
        sort_keys=False,
    )
    write_jsonl(run_dir / "tool_log.jsonl", [{"type": "edit", "path": "input.txt"}])
    (run_dir / "decision_trace.md").write_text("read edit verify", encoding="utf-8")
    (run_dir / "raw_response.txt").write_text('{"edits": {}}', encoding="utf-8")

    payload = build_dashboard_payload(run_dir, tmp_path / "runs")
    report = render_full_report(payload)

    assert payload["task_checks"][0]["passed"] is True
    assert payload["raw_response"]["valid_json_action"] is True
    assert payload["mechanistic"]["evidence"][0]["technique"] == "lora_delta_geometry"
    assert payload["mechanistic"]["deep_circuit_backends"]["sae_training"] == "unavailable"
    assert "Latest Best Model Report" in report
    assert "score.json" in report


def test_missing_partial_artifacts_do_not_crash(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "20260501T010102-trial-2"
    _write_run(run_dir, accepted=True, functionality=0.1, explainability=0.2)

    payload = build_dashboard_payload(run_dir, tmp_path / "runs")

    assert payload["task_checks"] == []
    assert payload["tool_log"] == []
    assert payload["mechanistic"]["available"] is False
    assert payload["raw_response"]["available"] is False


def test_http_dashboard_endpoints_return_html_and_json(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run_dir = root / "20260501T010103-trial-3"
    _write_run(run_dir, accepted=True, functionality=0.8, explainability=0.2)
    handler = _make_handler(root, None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        html_body = urllib.request.urlopen(f"{base_url}/", timeout=5).read().decode("utf-8")
        runs_payload = _read_url_json(f"{base_url}/api/runs")
        best_payload = _read_url_json(f"{base_url}/api/best")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert "Interpretability Dashboard" in html_body
    assert runs_payload["selected"]["path"] == str(run_dir)
    assert best_payload["run"]["path"] == str(run_dir)


def _read_url_json(url: str) -> dict[str, Any]:
    return json.loads(urllib.request.urlopen(url, timeout=5).read().decode("utf-8"))
