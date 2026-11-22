"""Optional Gemini live client. The API key is read from the environment only."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from teambench.models.base import ModelClient
from teambench.types import Message, ModelResponse, ToolCall, ToolSpec

DEFAULT_MODEL = "gemini-3-flash-preview"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiModel(ModelClient):
    """Live client for Gemini function calling.

    The key is read from GEMINI_API_KEY at call time and is never stored on the
    instance, logged, or written into results.
    """

    def __init__(self, model: str = DEFAULT_MODEL, timeout_s: float = 45.0) -> None:
        self.model = model
        self.name = model
        self.timeout_s = timeout_s

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        agent_name: str,
        agent_role: str,
        task_id: str,
        pattern: str,
        step: int,
        world: dict[str, object],
    ) -> ModelResponse:
        del agent_name, agent_role, task_id, pattern, step, world
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        body = _request_body(messages, tools)
        url = ENDPOINT.format(model=self.model) + "?key=" + key
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Do not include the request URL (it has the key) in the message.
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from None
        return _parse_response(raw)


def _request_body(messages: Sequence[Message], tools: Sequence[ToolSpec]) -> dict[str, Any]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
            continue
        if message.role == "tool":
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": message.tool_name or "tool",
                                "response": _parse_json(message.content),
                            }
                        }
                    ],
                }
            )
            continue
        parts: list[dict[str, Any]] = []
        if message.content:
            parts.append({"text": message.content})
        if message.tool_calls:
            for call in message.tool_calls:
                parts.append({"functionCall": {"name": call.name, "args": dict(call.arguments)}})
        if not parts:
            continue
        role = "model" if message.role == "assistant" else "user"
        contents.append({"role": role, "parts": parts})
    if not contents:
        contents.append({"role": "user", "parts": [{"text": "Begin."}]})
    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"temperature": 0, "maxOutputTokens": 2048},
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    if tools:
        payload["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.parameters),
                    }
                    for tool in tools
                ]
            }
        ]
    return payload


def _parse_json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"result": text}
    return value if isinstance(value, dict) else {"result": value}


def _parse_response(raw: MappingSafe) -> ModelResponse:
    usage = raw.get("usageMetadata") or {}
    prompt_tokens = int(usage.get("promptTokenCount") or 0)
    completion_tokens = int(usage.get("candidatesTokenCount") or usage.get("totalTokenCount") or 0)
    candidates = raw.get("candidates") or []
    if not candidates:
        return ModelResponse(content="", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    texts: list[str] = []
    calls: list[ToolCall] = []
    for i, part in enumerate(parts):
        if part.get("text"):
            texts.append(str(part["text"]))
        fn = part.get("functionCall")
        if fn:
            calls.append(
                ToolCall(
                    id=f"gemini-{i}",
                    name=str(fn.get("name") or ""),
                    arguments=dict(fn.get("args") or {}),
                )
            )
    return ModelResponse(
        content="\n".join(texts),
        tool_calls=calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=max(completion_tokens, 0),
    )


# Local alias so annotations stay simple without importing Mapping everywhere.
MappingSafe = dict[str, Any]
