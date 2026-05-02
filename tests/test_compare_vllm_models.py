from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml

import compare_vllm_models


class _CompareHandler(BaseHTTPRequestHandler):
    model_ids = ["base-model", "adapter-model"]
    request_models: list[str] = []

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json({"status": "ok"})
            return
        if self.path == "/v1/models":
            self._write_json({"data": [{"id": model_id} for model_id in self.model_ids]})
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        payload = json.loads(body)
        model = str(payload["model"])
        self.__class__.request_models.append(model)
        self._write_json(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision_trace": f"read edit test input.txt with {model}",
                                    "edits": {"input.txt": "DONE\n"},
                                    "commands": [["python3", "-c", "print('ok')"]],
                                }
                            )
                        }
                    }
                ]
            }
        )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _write_json(self, payload: dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))


class _FakeProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False
        self.waited = False
        self.returncode = None

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: int | None = None) -> int:
        self.waited = True
        return 0


@pytest.fixture()
def fake_server() -> str:
    _CompareHandler.model_ids = ["base-model", "adapter-model"]
    _CompareHandler.request_models = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CompareHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture()
def compare_config(tmp_path: Path, fake_server: str) -> Path:
    config = {
        "experiment": {"functionality_floor_ratio": 0.90},
        "model": {
            "backend": "vllm_openai",
            "name": "base-model",
            "adapter_name": "adapter-model",
            "adapter_path": "adapters/latest",
            "base_url": fake_server,
            "timeout_s": 5,
            "temperature": 0.0,
            "max_tokens": 256,
        },
        "resources": {"task_timeout_s": 10},
        "scoring": {
            "metrics_config": "configs/metrics.yaml",
            "task_paths": ["tasks/replace_token.yaml"],
        },
    }
    path = tmp_path / "vllm.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_compare_uses_base_and_lora_model_ids(compare_config: Path, tmp_path: Path) -> None:
    summary = compare_vllm_models.compare_models(
        config_path=compare_config,
        run_dir=tmp_path / "compare",
        skip_server_start=True,
        startup_timeout_s=2,
    )

    assert _CompareHandler.request_models == ["base-model", "adapter-model"]
    assert summary["base_model"]["model_id"] == "base-model"
    assert summary["lora_model"]["model_id"] == "adapter-model"
    assert summary["tasks"][0]["base"]["functionality"] == 1.0
    assert summary["tasks"][0]["lora"]["functionality"] == 1.0


def test_missing_model_ids_fail_clearly(compare_config: Path, tmp_path: Path) -> None:
    _CompareHandler.model_ids = ["base-model"]

    with pytest.raises(SystemExit, match="adapter-model"):
        compare_vllm_models.compare_models(
            config_path=compare_config,
            run_dir=tmp_path / "compare",
            skip_server_start=True,
            startup_timeout_s=2,
        )


def test_reports_include_both_models_and_score_details(compare_config: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "compare"

    summary = compare_vllm_models.compare_models(
        config_path=compare_config,
        run_dir=run_dir,
        skip_server_start=True,
        startup_timeout_s=2,
    )

    saved_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert saved_summary == summary
    assert "base-model" in report
    assert "adapter-model" in report
    assert "behavior_trace_faithfulness" in saved_summary["tasks"][0]["base"]["details"]
    assert (run_dir / "base" / "replace_token" / "score.json").exists()
    assert (run_dir / "lora" / "replace_token" / "score.json").exists()


def test_server_cleanup_runs_on_failure(compare_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_process = _FakeProcess()
    config = yaml.safe_load(compare_config.read_text(encoding="utf-8"))
    config["model"]["base_url"] = "http://127.0.0.1:9"
    blocked_config = tmp_path / "blocked-vllm.yaml"
    blocked_config.write_text(yaml.safe_dump(config), encoding="utf-8")

    def fake_popen(*args: object, **kwargs: object) -> _FakeProcess:
        return fake_process

    monkeypatch.setattr(compare_vllm_models.subprocess, "Popen", fake_popen)
    _CompareHandler.model_ids = ["base-model"]

    with pytest.raises(SystemExit):
        compare_vllm_models.compare_models(
            config_path=blocked_config,
            run_dir=tmp_path / "compare",
            skip_server_start=False,
            startup_timeout_s=0.1,
        )

    assert fake_process.terminated
    assert fake_process.waited


def test_starting_server_fails_when_configured_port_is_busy(compare_config: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="already in use"):
        compare_vllm_models.compare_models(
            config_path=compare_config,
            run_dir=tmp_path / "compare",
            skip_server_start=False,
            startup_timeout_s=2,
        )


def test_wait_for_health_fails_when_started_process_exits() -> None:
    class ExitedProcess:
        returncode = 98

        def poll(self) -> int:
            return self.returncode

    with pytest.raises(SystemExit, match="exited before becoming healthy with code 98"):
        compare_vllm_models.wait_for_health("http://127.0.0.1:9", 2, ExitedProcess())  # type: ignore[arg-type]
