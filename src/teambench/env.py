"""In-process tool sandbox for a single trial."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from teambench.types import ToolSpec


@dataclass
class ToolResult:
    ok: bool
    data: Any
    error: str | None = None

    def render(self) -> str:
        payload: dict[str, Any] = {"ok": self.ok, "data": self.data}
        if self.error:
            payload["error"] = self.error
        return json.dumps(payload, default=str, sort_keys=True)


ToolHandler = Callable[[dict[str, Any], dict[str, Any], str], ToolResult]


@dataclass
class ToolEnv:
    """Mutable world plus named tools. One instance per trial."""

    world: dict[str, Any] = field(default_factory=dict)
    tools: dict[str, tuple[ToolSpec, ToolHandler]] = field(default_factory=dict)
    log: list[dict[str, Any]] = field(default_factory=list)
    blocked: list[dict[str, str]] = field(default_factory=list)

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        self.tools[spec.name] = (spec, handler)

    def specs(self, names: Sequence[str] | None = None) -> list[ToolSpec]:
        if names is None:
            return [pair[0] for pair in self.tools.values()]
        return [self.tools[name][0] for name in names if name in self.tools]

    def invoke(
        self,
        name: str,
        arguments: Mapping[str, Any],
        caller: str,
        allowlist: Sequence[str],
    ) -> ToolResult:
        allowed = set(allowlist)
        args = dict(arguments)
        if name not in allowed or name not in self.tools:
            result = ToolResult(
                ok=False,
                data=None,
                error=f"tool '{name}' is not available to agent '{caller}'",
            )
            self.blocked.append({"agent": caller, "tool": name})
            self.log.append({"agent": caller, "tool": name, "ok": False, "blocked": True, "args": args})
            return result
        handler = self.tools[name][1]
        try:
            result = handler(args, self.world, caller)
        except Exception as exc:
            result = ToolResult(ok=False, data=None, error=f"tool raised: {exc}")
        self.log.append(
            {
                "agent": caller,
                "tool": name,
                "ok": result.ok,
                "blocked": False,
                "args": args,
                "error": result.error,
            }
        )
        return result


def object_schema(properties: Mapping[str, Any], required: Sequence[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": dict(properties),
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


def make_tool(
    name: str,
    description: str,
    properties: Mapping[str, Any],
    required: Sequence[str] | None,
    handler: ToolHandler,
) -> tuple[ToolSpec, ToolHandler]:
    return ToolSpec(name, description, object_schema(properties, required)), handler
