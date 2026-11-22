from __future__ import annotations

import json
from typing import Any

import pytest

from teambench.models import gemini as gemini_mod
from teambench.models.gemini import GeminiModel, _parse_response, _request_body
from teambench.types import Message, ToolCall, ToolSpec


class _FakeResp:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_request_body_maps_roles_and_tools() -> None:
    tools = [ToolSpec("kb.get", "Get a fact.", {"type": "object", "properties": {"topic": {"type": "string"}}})]
    messages = [
        Message(role="system", content="Be careful."),
        Message(role="user", content="Go."),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="1", name="kb.get", arguments={"topic": "alpha"})],
        ),
        Message(role="tool", content='{"ok": true, "data": {"year": 2019}}', tool_name="kb.get"),
    ]
    body = _request_body(messages, tools)
    assert body["systemInstruction"]["parts"][0]["text"] == "Be careful."
    assert body["contents"][0]["role"] == "user"
    assert body["contents"][1]["role"] == "model"
    assert "functionCall" in body["contents"][1]["parts"][0]
    assert "functionResponse" in body["contents"][2]["parts"][0]
    assert body["tools"][0]["functionDeclarations"][0]["name"] == "kb.get"


def test_parse_response_reads_function_calls_and_usage() -> None:
    raw = {
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7},
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "ok"},
                        {"functionCall": {"name": "submit", "args": {"payload": "done"}}},
                    ]
                }
            }
        ],
    }
    response = _parse_response(raw)
    assert response.content == "ok"
    assert response.tool_calls[0].name == "submit"
    assert response.prompt_tokens == 11
    assert response.completion_tokens == 7


def test_complete_uses_env_key_and_hides_it(monkeypatch, tmp_path) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: object, timeout: float = 0) -> _FakeResp:
        captured["full_url"] = request.full_url  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return _FakeResp(
            {
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1},
                "candidates": [{"content": {"parts": [{"text": "hi"}]}}],
            }
        )

    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-key-not-real")
    monkeypatch.setattr(gemini_mod.urllib.request, "urlopen", fake_urlopen)
    model = GeminiModel(timeout_s=3)
    result = model.complete(
        [Message(role="user", content="Hello")],
        [],
        agent_name="solo",
        agent_role="solo",
        task_id="T01",
        pattern="single",
        step=0,
        world={},
    )
    assert result.content == "hi"
    assert "secret-test-key-not-real" in captured["full_url"]
    # The client must not stash the key on the instance.
    assert "secret-test-key-not-real" not in repr(model) + json.dumps(model.__dict__, default=str)


def test_complete_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiModel().complete(
            [Message(role="user", content="x")],
            [],
            agent_name="solo",
            agent_role="solo",
            task_id="T01",
            pattern="single",
            step=0,
            world={},
        )
