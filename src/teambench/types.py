"""Shared types for TeamBench."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Pattern(str, Enum):
    SINGLE = "single"
    SUPERVISOR = "supervisor_worker"
    ROUND_ROBIN = "round_robin"
    DEBATE = "debate"


class Role(str, Enum):
    SOLO = "solo"
    SUPERVISOR = "supervisor"
    WORKER = "worker"
    JUDGE = "judge"
    DEBATER = "debater"


class FailureTag(str, Enum):
    GOAL_MISS = "goal_miss"
    CONSTRAINT_VIOLATION = "constraint_violation"
    HANDOFF_LOSS = "handoff_loss"
    TOOL_ERROR_UNRECOVERED = "tool_error_unrecovered"
    INJECTION_FOLLOWED = "injection_followed"
    CONFLICT_WRONG_RESOLUTION = "conflict_wrong_resolution"
    STEP_BUDGET_EXCEEDED = "step_budget_exceeded"
    POLICY_BLOCKED_TOOL = "policy_blocked_tool"
    INCONSISTENT_STATE = "inconsistent_state"
    MODEL_ERROR = "model_error"


PATTERNS: tuple[Pattern, ...] = tuple(Pattern)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class Message:
    role: str  # system, user, assistant, tool
    content: str
    name: str | None = None
    tool_calls: Sequence[ToolCall] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]

    def json_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: Role
    system_prompt: str
    tools: Sequence[str]


@dataclass(frozen=True)
class ModelResponse:
    content: str = ""
    tool_calls: Sequence[ToolCall] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Grade:
    passed: bool
    failures: list[FailureTag] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    task_id: str
    pattern: str
    trial: int
    seed: int
    success: bool
    failures: list[str]
    steps: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    elapsed_ms: int
    reasons: list[str]
    submitted: Any = None
    model: str = "mock"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "pattern": self.pattern,
            "trial": self.trial,
            "seed": self.seed,
            "success": self.success,
            "failures": list(self.failures),
            "steps": self.steps,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "elapsed_ms": self.elapsed_ms,
            "reasons": list(self.reasons),
            "submitted": self.submitted,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrialResult:
        return cls(
            task_id=str(data["task_id"]),
            pattern=str(data["pattern"]),
            trial=int(data["trial"]),
            seed=int(data["seed"]),
            success=bool(data["success"]),
            failures=[str(x) for x in data.get("failures", [])],
            steps=int(data.get("steps", 0)),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
            elapsed_ms=int(data.get("elapsed_ms", 0)),
            reasons=[str(x) for x in data.get("reasons", [])],
            submitted=data.get("submitted"),
            model=str(data.get("model", "mock")),
        )


@dataclass
class RunManifest:
    version: str
    created_at: str
    model: str
    seed: int
    repeats: int
    fallibility: float
    patterns: list[str]
    task_ids: list[str]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "model": self.model,
            "seed": self.seed,
            "repeats": self.repeats,
            "fallibility": self.fallibility,
            "patterns": list(self.patterns),
            "task_ids": list(self.task_ids),
            "notes": self.notes,
        }
