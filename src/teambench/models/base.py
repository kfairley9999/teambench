"""Model client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from teambench.types import Message, ModelResponse, ToolSpec


class ModelClient(ABC):
    """A complete() call is one billed model step."""

    name: str = "base"

    @abstractmethod
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
        raise NotImplementedError
