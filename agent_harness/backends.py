from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class AgentAction:
    decision_trace: str
    edits: dict[str, str]
    commands: list[list[str]]
    raw_response: str | None = None
    parsed_action: dict[str, Any] = field(default_factory=dict)


class ModelBackend:
    def propose_actions(self, instruction: str, work_dir: Path) -> AgentAction:
        raise NotImplementedError


class BackendError(RuntimeError):
    """Raised when a model backend cannot produce a valid action."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
        extracted_response: str | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.extracted_response = extracted_response


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
        use_response_format: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = adapter_name or model_name
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_response_format = use_response_format

    def propose_actions(self, instruction: str, work_dir: Path) -> AgentAction:
        workspace_snapshot = _workspace_snapshot(work_dir)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a coding task agent. Return one JSON object and no other text. "
                        "The JSON object must have exactly these keys: decision_trace, edits, "
                        "and commands. decision_trace must be a non-empty string. edits must map "
                        "file paths from the Workspace files headers to full replacement file "
                        "contents. Do not include the working directory in edit paths. commands "
                        "must be a list of argv arrays. Do not include markdown fences."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task instruction:\n{instruction}\n\n"
                        f"Working directory: {work_dir}\n"
                        f"Workspace files:\n{workspace_snapshot}\n\n"
                        "Use edit paths exactly as shown after each `---` in Workspace files, "
                        "for example `input.txt`, not the full working directory path.\n"
                        "Produce the next complete action as JSON. Example shape:\n"
                        '{"decision_trace":"...","edits":{"relative/path.txt":"full file contents"},'
                        '"commands":[["python3","-c","..."]]}'
                    ),
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.use_response_format:
            payload["response_format"] = {"type": "json_object"}
        data = self._post_chat_completions(payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError("vLLM response did not contain choices[0].message.content") from exc
        if not isinstance(content, str):
            raise BackendError("vLLM response content was not a string")
        return _agent_action_from_json(content)

    def _post_chat_completions(
        self,
        payload: dict[str, Any],
        *,
        retry_without_response_format: bool = True,
    ) -> dict[str, Any]:
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
            if retry_without_response_format and "response_format" in payload and exc.code in {400, 500}:
                fallback_payload = dict(payload)
                fallback_payload.pop("response_format", None)
                try:
                    return self._post_chat_completions(
                        fallback_payload,
                        retry_without_response_format=False,
                    )
                except BackendError as fallback_exc:
                    detail = f": {error_body}" if error_body else ""
                    raise BackendError(
                        "vLLM OpenAI server rejected /v1/chat/completions "
                        f"with HTTP {exc.code} {exc.reason}{detail}; retry without "
                        "response_format also failed. A server-side HTTP 500 while "
                        "using response_format can terminate vLLM before this client "
                        f"fallback can recover: {fallback_exc}"
                    ) from exc
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
    raw_content = content
    content = _extract_json_object(content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        preview = _content_preview(content)
        raise BackendError(
            f"Model response was not strict JSON. Response preview: {preview}",
            raw_response=raw_content,
            extracted_response=content,
        ) from exc
    if not isinstance(data, dict):
        raise BackendError("Model response JSON must be an object", raw_response=raw_content, extracted_response=content)
    decision_trace = data.get("decision_trace")
    edits = data.get("edits")
    commands = data.get("commands")
    if not isinstance(decision_trace, str) or not decision_trace.strip():
        raise BackendError(
            "Model response must include a non-empty string decision_trace",
            raw_response=raw_content,
            extracted_response=content,
        )
    if not isinstance(edits, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in edits.items()):
        raise BackendError(
            "Model response edits must be an object mapping paths to strings",
            raw_response=raw_content,
            extracted_response=content,
        )
    if not isinstance(commands, list) or not all(
        isinstance(command, list) and all(isinstance(part, str) for part in command) for command in commands
    ):
        raise BackendError(
            "Model response commands must be a list of string argv arrays",
            raw_response=raw_content,
            extracted_response=content,
        )
    return AgentAction(decision_trace=decision_trace, edits=edits, commands=commands, raw_response=raw_content, parsed_action=data)


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
    if stripped.startswith("{"):
        return stripped

    start = stripped.find("{")
    if start < 0:
        return stripped
    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(stripped[start:])
    except json.JSONDecodeError:
        return stripped
    return stripped[start : start + end]


def _content_preview(content: str, limit: int = 240) -> str:
    compact = " ".join(content.strip().split())
    if len(compact) > limit:
        compact = compact[: limit - 3] + "..."
    return repr(compact)


def _workspace_snapshot(work_dir: Path, *, max_files: int = 20, max_bytes_per_file: int = 4000) -> str:
    if not work_dir.exists():
        return "(working directory does not exist)"

    chunks: list[str] = []
    files = sorted(path for path in work_dir.rglob("*") if path.is_file())
    for path in files[:max_files]:
        rel_path = path.relative_to(work_dir).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            chunks.append(f"--- {rel_path} ---\n(binary file omitted)")
            continue
        if len(content.encode("utf-8")) > max_bytes_per_file:
            content = content[:max_bytes_per_file] + "\n...[truncated]"
        chunks.append(f"--- {rel_path} ---\n{content}")
    if len(files) > max_files:
        chunks.append(f"...[{len(files) - max_files} more files omitted]")
    return "\n\n".join(chunks) if chunks else "(no files)"


def validate_model_config(config: dict[str, Any]) -> dict[str, Any]:
    backend = str(config.get("backend", "mock"))
    try:
        return _BACKEND_REGISTRY[backend].validator(config)
    except KeyError as exc:
        raise NotImplementedError(f"Backend {backend!r} is not implemented. Use 'mock' or 'vllm_openai'.") from exc


def make_backend(config: dict) -> ModelBackend:
    validated = validate_model_config(config)
    return _BACKEND_REGISTRY[validated["backend"]].factory(validated)


@dataclass(frozen=True)
class BackendSpec:
    validator: Callable[[dict[str, Any]], dict[str, Any]]
    factory: Callable[[dict[str, Any]], ModelBackend]


def _validate_mock(config: dict[str, Any]) -> dict[str, Any]:
    return {"backend": str(config.get("backend", "mock"))}


def _validate_vllm_openai(config: dict[str, Any]) -> dict[str, Any]:
    adapter_name = config.get("adapter_name")
    adapter_path = config.get("adapter_path")
    max_tokens = int(config.get("max_tokens", 1024))
    raw_max_model_len = config.get("max_model_len")
    max_model_len = int(raw_max_model_len) if raw_max_model_len is not None else None
    _validate_optional_nonempty(adapter_name, "model.adapter_name")
    _validate_optional_nonempty(adapter_path, "model.adapter_path")
    if adapter_name is not None and adapter_path is None:
        raise ValueError("model.adapter_path must be a non-empty string when model.adapter_name is configured")
    if max_model_len is not None and max_tokens >= max_model_len:
        raise ValueError(
            "model.max_tokens must be less than model.max_model_len because vLLM requires "
            "prompt_tokens + max_tokens <= max_model_len"
        )
    return {
        "backend": "vllm_openai",
        "base_url": str(config.get("base_url", "http://127.0.0.1:8000")),
        "name": str(config.get("name", "Qwen/Qwen2.5-Coder-1.5B-Instruct")),
        "adapter_name": adapter_name,
        "timeout_s": float(config.get("timeout_s", 30)),
        "temperature": float(config.get("temperature", 0.0)),
        "max_tokens": max_tokens,
        "max_model_len": max_model_len,
        "use_response_format": bool(config.get("use_response_format", True)),
    }


def _validate_optional_nonempty(value: Any, label: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{label} must be a non-empty string when provided")


def _make_vllm_openai(config: dict[str, Any]) -> ModelBackend:
    return VLLMOpenAIBackend(
        base_url=config["base_url"],
        model_name=config["name"],
        adapter_name=config["adapter_name"],
        timeout_s=config["timeout_s"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        use_response_format=config["use_response_format"],
    )


_BACKEND_REGISTRY: dict[str, BackendSpec] = {
    "mock": BackendSpec(_validate_mock, lambda config: MockBackend()),
    "vllm_openai": BackendSpec(_validate_vllm_openai, _make_vllm_openai),
}
