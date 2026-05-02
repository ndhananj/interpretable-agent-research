from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agent_harness.backends import BackendError, VLLMOpenAIBackend, make_backend


class _Handler(BaseHTTPRequestHandler):
    response_content = {
        "decision_trace": "edit the file",
        "edits": {"input.txt": "DONE\n"},
        "commands": [["python3", "-c", "print('ok')"]],
    }

    def do_POST(self) -> None:  # noqa: N802
        _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(self.response_content),
                            }
                        }
                    ]
                }
            ).encode("utf-8")
        )

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture()
def fake_server() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_vllm_response_becomes_agent_action(fake_server: str, tmp_path: Path) -> None:
    backend = VLLMOpenAIBackend(base_url=fake_server, model_name="base", adapter_name="adapter")

    action = backend.propose_actions("replace token", tmp_path)

    assert action.decision_trace == "edit the file"
    assert action.edits == {"input.txt": "DONE\n"}
    assert action.commands == [["python3", "-c", "print('ok')"]]


def test_vllm_unreachable_server_raises_clear_error(tmp_path: Path) -> None:
    backend = VLLMOpenAIBackend(base_url="http://127.0.0.1:9", model_name="base", timeout_s=0.2)

    with pytest.raises(BackendError, match="Could not reach local vLLM OpenAI server"):
        backend.propose_actions("replace token", tmp_path)


def test_vllm_invalid_model_json_is_rejected(fake_server: str, tmp_path: Path) -> None:
    original = _Handler.response_content
    _Handler.response_content = {"decision_trace": "missing fields"}
    try:
        backend = VLLMOpenAIBackend(base_url=fake_server, model_name="base")
        with pytest.raises(BackendError, match="edits"):
            backend.propose_actions("replace token", tmp_path)
    finally:
        _Handler.response_content = original


def test_make_backend_creates_vllm_without_network_call() -> None:
    backend = make_backend({"backend": "vllm_openai", "base_url": "http://127.0.0.1:8000"})
    assert isinstance(backend, VLLMOpenAIBackend)
