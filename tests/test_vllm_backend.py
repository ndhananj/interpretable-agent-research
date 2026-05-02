from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agent_harness.backends import BackendError, VLLMOpenAIBackend, make_backend


class _Handler(BaseHTTPRequestHandler):
    response_status = 200
    response_body = None
    response_content = {
        "decision_trace": "edit the file",
        "edits": {"input.txt": "DONE\n"},
        "commands": [["python3", "-c", "print('ok')"]],
    }
    request_payload = {}

    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.__class__.request_payload = json.loads(body)
        self.send_response(self.response_status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response_body = self.response_body
        if response_body is None:
            response_body = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(self.response_content),
                        }
                    }
                ]
            }
        self.wfile.write(json.dumps(response_body).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture()
def fake_server() -> str:
    _Handler.response_status = 200
    _Handler.response_body = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        _Handler.response_status = 200
        _Handler.response_body = None


def test_vllm_response_becomes_agent_action(fake_server: str, tmp_path: Path) -> None:
    backend = VLLMOpenAIBackend(base_url=fake_server, model_name="base", adapter_name="adapter")

    action = backend.propose_actions("replace token", tmp_path)

    assert _Handler.request_payload["model"] == "adapter"
    assert action.decision_trace == "edit the file"
    assert action.edits == {"input.txt": "DONE\n"}
    assert action.commands == [["python3", "-c", "print('ok')"]]


def test_vllm_unreachable_server_raises_clear_error(tmp_path: Path) -> None:
    backend = VLLMOpenAIBackend(base_url="http://127.0.0.1:9", model_name="base", timeout_s=0.2)

    with pytest.raises(BackendError, match="Could not reach local vLLM OpenAI server"):
        backend.propose_actions("replace token", tmp_path)


def test_vllm_http_error_includes_status_and_server_body(fake_server: str, tmp_path: Path) -> None:
    _Handler.response_status = 400
    _Handler.response_body = {
        "error": {
            "message": "response_format json_object is not supported by this model",
            "type": "BadRequestError",
        }
    }
    backend = VLLMOpenAIBackend(base_url=fake_server, model_name="base")

    with pytest.raises(BackendError) as exc_info:
        backend.propose_actions("replace token", tmp_path)

    message = str(exc_info.value)
    assert "HTTP 400" in message
    assert "response_format json_object is not supported by this model" in message


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


def test_make_backend_requires_adapter_path_with_adapter_name() -> None:
    with pytest.raises(ValueError, match="adapter_path"):
        make_backend({"backend": "vllm_openai", "adapter_name": "adapter"})
