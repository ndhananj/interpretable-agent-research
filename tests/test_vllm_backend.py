from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agent_harness.backends import BackendError, VLLMOpenAIBackend, make_backend, validate_model_config


class _Handler(BaseHTTPRequestHandler):
    response_status = 200
    response_body = None
    response_content = {
        "decision_trace": "edit the file",
        "edits": {"input.txt": "DONE\n"},
        "commands": [["python3", "-c", "print('ok')"]],
    }
    request_payload = {}
    request_payloads = []
    fail_response_format_once = False

    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.__class__.request_payload = json.loads(body)
        self.__class__.request_payloads.append(self.__class__.request_payload)
        if self.__class__.fail_response_format_once and "response_format" in self.__class__.request_payload:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "error": {
                            "message": "response_format json_object is not supported by this model",
                            "type": "InternalServerError",
                        }
                    }
                ).encode("utf-8")
            )
            return
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
    _Handler.request_payload = {}
    _Handler.request_payloads = []
    _Handler.fail_response_format_once = False
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
        _Handler.request_payload = {}
        _Handler.request_payloads = []
        _Handler.fail_response_format_once = False


def test_vllm_response_becomes_agent_action(fake_server: str, tmp_path: Path) -> None:
    (tmp_path / "input.txt").write_text("TODO\n", encoding="utf-8")
    backend = VLLMOpenAIBackend(base_url=fake_server, model_name="base", adapter_name="adapter")

    action = backend.propose_actions("replace token", tmp_path)

    assert _Handler.request_payload["model"] == "adapter"
    assert "--- input.txt ---\nTODO\n" in _Handler.request_payload["messages"][1]["content"]
    assert action.decision_trace == "edit the file"
    assert action.edits == {"input.txt": "DONE\n"}
    assert action.commands == [["python3", "-c", "print('ok')"]]
    assert action.raw_response is not None
    assert action.parsed_action["edits"] == {"input.txt": "DONE\n"}


def test_vllm_retries_without_response_format_when_json_mode_fails(fake_server: str, tmp_path: Path) -> None:
    _Handler.fail_response_format_once = True
    backend = VLLMOpenAIBackend(base_url=fake_server, model_name="base")

    action = backend.propose_actions("replace token", tmp_path)

    assert action.decision_trace == "edit the file"
    assert len(_Handler.request_payloads) == 2
    assert _Handler.request_payloads[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in _Handler.request_payloads[1]


def test_vllm_can_disable_response_format(fake_server: str, tmp_path: Path) -> None:
    backend = VLLMOpenAIBackend(base_url=fake_server, model_name="base", use_response_format=False)

    backend.propose_actions("replace token", tmp_path)

    assert len(_Handler.request_payloads) == 1
    assert "response_format" not in _Handler.request_payload


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


def test_vllm_accepts_json_wrapped_in_markdown_fence(fake_server: str, tmp_path: Path) -> None:
    original = _Handler.response_body
    content = json.dumps(_Handler.response_content)
    _Handler.response_body = {"choices": [{"message": {"content": f"```json\n{content}\n```"}}]}
    try:
        backend = VLLMOpenAIBackend(base_url=fake_server, model_name="base", use_response_format=False)
        action = backend.propose_actions("replace token", tmp_path)
    finally:
        _Handler.response_body = original

    assert action.edits == {"input.txt": "DONE\n"}


def test_vllm_accepts_json_after_leading_text(fake_server: str, tmp_path: Path) -> None:
    original = _Handler.response_body
    content = json.dumps(_Handler.response_content)
    _Handler.response_body = {"choices": [{"message": {"content": f"Here is the JSON:\n{content}"}}]}
    try:
        backend = VLLMOpenAIBackend(base_url=fake_server, model_name="base", use_response_format=False)
        action = backend.propose_actions("replace token", tmp_path)
    finally:
        _Handler.response_body = original

    assert action.commands == [["python3", "-c", "print('ok')"]]


def test_vllm_non_json_error_includes_response_preview(fake_server: str, tmp_path: Path) -> None:
    original = _Handler.response_body
    _Handler.response_body = {"choices": [{"message": {"content": "I would edit input.txt."}}]}
    try:
        backend = VLLMOpenAIBackend(base_url=fake_server, model_name="base", use_response_format=False)
        with pytest.raises(BackendError, match="Response preview: 'I would edit input.txt.'"):
            backend.propose_actions("replace token", tmp_path)
    finally:
        _Handler.response_body = original


def test_vllm_backend_error_preserves_malformed_raw_response(fake_server: str, tmp_path: Path) -> None:
    original = _Handler.response_body
    _Handler.response_body = {"choices": [{"message": {"content": '{"decision_trace":'}}]}
    try:
        backend = VLLMOpenAIBackend(base_url=fake_server, model_name="base", use_response_format=False)
        with pytest.raises(BackendError) as exc_info:
            backend.propose_actions("replace token", tmp_path)
    finally:
        _Handler.response_body = original

    assert exc_info.value.raw_response == '{"decision_trace":'
    assert exc_info.value.extracted_response == '{"decision_trace":'


def test_vllm_backend_error_preserves_non_json_raw_response(fake_server: str, tmp_path: Path) -> None:
    original = _Handler.response_body
    _Handler.response_body = {"choices": [{"message": {"content": "I would edit input.txt."}}]}
    try:
        backend = VLLMOpenAIBackend(base_url=fake_server, model_name="base", use_response_format=False)
        with pytest.raises(BackendError) as exc_info:
            backend.propose_actions("replace token", tmp_path)
    finally:
        _Handler.response_body = original

    assert exc_info.value.raw_response == "I would edit input.txt."


def test_make_backend_creates_vllm_without_network_call() -> None:
    backend = make_backend({"backend": "vllm_openai", "base_url": "http://127.0.0.1:8000"})
    assert isinstance(backend, VLLMOpenAIBackend)


def test_make_backend_requires_adapter_path_with_adapter_name() -> None:
    with pytest.raises(ValueError, match="adapter_path"):
        make_backend({"backend": "vllm_openai", "adapter_name": "adapter"})


def test_vllm_config_rejects_completion_budget_equal_to_context_window() -> None:
    with pytest.raises(ValueError, match=r"prompt_tokens \+ max_tokens <= max_model_len"):
        validate_model_config(
            {
                "backend": "vllm_openai",
                "max_model_len": 1024,
                "max_tokens": 1024,
            }
        )


def test_vllm_config_accepts_completion_budget_below_context_window() -> None:
    config = validate_model_config(
        {
            "backend": "vllm_openai",
            "max_model_len": 1024,
            "max_tokens": 512,
        }
    )

    assert config["max_model_len"] == 1024
    assert config["max_tokens"] == 512


def test_vllm_config_can_disable_response_format() -> None:
    config = validate_model_config(
        {
            "backend": "vllm_openai",
            "use_response_format": False,
        }
    )

    assert config["use_response_format"] is False
