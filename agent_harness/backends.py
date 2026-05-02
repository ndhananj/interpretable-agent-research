from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentAction:
    decision_trace: str
    edits: dict[str, str]
    commands: list[list[str]]


class ModelBackend:
    def propose_actions(self, instruction: str, work_dir: Path) -> AgentAction:
        raise NotImplementedError


class BackendError(RuntimeError):
    """Raised when a model backend cannot produce a valid action."""


class MockBackend(ModelBackend):
    """Deterministic backend for offline smoke tests."""

    def propose_actions(self, instruction: str, work_dir: Path) -> AgentAction:
        edits: dict[str, str] = {}
        input_path = work_dir / "input.txt"
        if input_path.exists():
            text = input_path.read_text(encoding="utf-8")
            edits["input.txt"] = text.replace("TODO", "DONE")
        return AgentAction(
            decision_trace=(
                "I read the task file, edit input.txt to replace the target token, "
                "and test the result with a small python3 command."
            ),
            edits=edits,
            commands=[["python3", "-c", "from pathlib import Path; assert 'DONE' in Path('input.txt').read_text()"]],
        )


class VLLMOpenAIBackend(ModelBackend):
    """OpenAI-compatible chat backend for a local vLLM server."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        adapter_name: str | None = None,
        timeout_s: float = 30.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = adapter_name or model_name
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens

    def propose_actions(self, instruction: str, work_dir: Path) -> AgentAction:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a coding task agent. Return only strict JSON with keys "
                        "decision_trace, edits, and commands. edits must map relative file "
                        "paths to full replacement file contents. commands must be a list "
                        "of argv arrays. Do not include markdown fences."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task instruction:\n{instruction}\n\n"
                        f"Working directory: {work_dir}\n"
                        "Produce the next complete action as JSON."
                    ),
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        data = self._post_chat_completions(payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError("vLLM response did not contain choices[0].message.content") from exc
        if not isinstance(content, str):
            raise BackendError("vLLM response content was not a string")
        return _agent_action_from_json(content)

    def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            detail = f": {error_body}" if error_body else ""
            raise BackendError(
                "vLLM OpenAI server rejected /v1/chat/completions "
                f"with HTTP {exc.code} {exc.reason}{detail}"
            ) from exc
        except (ConnectionError, TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            raise BackendError(
                f"Could not reach local vLLM OpenAI server at {self.base_url}. "
                "Start it with `vllm serve ... --host 127.0.0.1 --port 8000` "
                "or update model.base_url."
            ) from exc
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise BackendError("vLLM server returned non-JSON response") from exc
        if not isinstance(parsed, dict):
            raise BackendError("vLLM server returned a JSON value that was not an object")
        return parsed


def _agent_action_from_json(content: str) -> AgentAction:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise BackendError("Model response was not strict JSON") from exc
    if not isinstance(data, dict):
        raise BackendError("Model response JSON must be an object")
    decision_trace = data.get("decision_trace")
    edits = data.get("edits")
    commands = data.get("commands")
    if not isinstance(decision_trace, str) or not decision_trace.strip():
        raise BackendError("Model response must include a non-empty string decision_trace")
    if not isinstance(edits, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in edits.items()):
        raise BackendError("Model response edits must be an object mapping paths to strings")
    if not isinstance(commands, list) or not all(
        isinstance(command, list) and all(isinstance(part, str) for part in command) for command in commands
    ):
        raise BackendError("Model response commands must be a list of string argv arrays")
    return AgentAction(decision_trace=decision_trace, edits=edits, commands=commands)


def validate_model_config(config: dict[str, Any]) -> dict[str, Any]:
    backend = str(config.get("backend", "mock"))
    if backend == "mock":
        return {"backend": backend}
    if backend == "vllm_openai":
        adapter_name = config.get("adapter_name")
        adapter_path = config.get("adapter_path")
        if adapter_name is not None and (not isinstance(adapter_name, str) or not adapter_name.strip()):
            raise ValueError("model.adapter_name must be a non-empty string when provided")
        if adapter_name is not None and (not isinstance(adapter_path, str) or not adapter_path.strip()):
            raise ValueError(
                "model.adapter_path must be a non-empty string when model.adapter_name is configured"
            )
        if adapter_path is not None and (not isinstance(adapter_path, str) or not adapter_path.strip()):
            raise ValueError("model.adapter_path must be a non-empty string when provided")
        return {
            "backend": backend,
            "base_url": str(config.get("base_url", "http://127.0.0.1:8000")),
            "name": str(config.get("name", "Qwen/Qwen2.5-Coder-1.5B-Instruct")),
            "adapter_name": adapter_name,
            "timeout_s": float(config.get("timeout_s", 30)),
            "temperature": float(config.get("temperature", 0.0)),
            "max_tokens": int(config.get("max_tokens", 1024)),
        }
    raise NotImplementedError(
        f"Backend {backend!r} is not implemented. Use 'mock' or 'vllm_openai'."
    )


def make_backend(config: dict) -> ModelBackend:
    validated = validate_model_config(config)
    backend = validated["backend"]
    if backend == "mock":
        return MockBackend()
    if backend == "vllm_openai":
        return VLLMOpenAIBackend(
            base_url=validated["base_url"],
            model_name=validated["name"],
            adapter_name=validated["adapter_name"],
            timeout_s=validated["timeout_s"],
            temperature=validated["temperature"],
            max_tokens=validated["max_tokens"],
        )
    raise NotImplementedError(
        f"Backend {backend!r} is not implemented. Use 'mock' or 'vllm_openai'."
    )
