"""Scripted, offline model used as the published baseline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random

from teambench.models.base import ModelClient
from teambench.models.policies import MockContext, act
from teambench.types import Message, ModelResponse, ToolSpec


@dataclass
class ScriptedModel(ModelClient):
    """Deterministic role-conditioned policy with optional seeded fallibility.

    Fallibility is decided once per trial (stored on the world as `_trial_slip`).
    When a trial slips, policy-aware roles behave like gullible workers for that
    trial only. Same seed and trial index always produce the same slip bit.
    """

    fallibility: float = 0.0
    name: str = "mock"

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
        raw_seed = world.get("_trial_seed", 0)
        seed = int(raw_seed) if isinstance(raw_seed, (int, float, str)) else 0
        rng = Random(seed * 1_000_003 + step * 13 + sum(map(ord, agent_name)))
        if "_trial_slip" not in world:
            slip_rng = Random(seed * 17 + 29)
            world["_trial_slip"] = bool(slip_rng.random() < self.fallibility)
        slip = bool(world["_trial_slip"])
        ctx = MockContext(
            task_id=task_id,
            pattern=pattern,
            agent_name=agent_name,
            agent_role=agent_role,
            messages=list(messages),
            tool_names=tuple(t.name for t in tools),
            world=world,
            step=step,
            rng=rng,
            slip=slip,
        )
        return act(ctx)
