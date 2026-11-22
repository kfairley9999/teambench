"""Task registry."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from teambench.env import ToolHandler
from teambench.types import AgentSpec, Grade, Pattern, ToolSpec

SetupFn = Callable[[], dict[str, Any]]
ToolsFn = Callable[[], list[tuple[ToolSpec, ToolHandler]]]
GradeFn = Callable[[dict[str, Any]], Grade]
TeamFn = Callable[[Pattern], list[AgentSpec]]


@dataclass(frozen=True)
class Task:
    id: str
    name: str
    category: str
    description: str
    goal: str
    max_steps: int
    setup: SetupFn
    tools: ToolsFn
    grade: GradeFn
    team: TeamFn
    categories: tuple[str, ...] = field(default=())

    def domain_tool_names(self) -> list[str]:
        return [spec.name for spec, _ in self.tools()]


_REGISTRY: dict[str, Task] = {}


def register(task: Task) -> Task:
    if task.id in _REGISTRY:
        raise ValueError(f"duplicate task id: {task.id}")
    _REGISTRY[task.id] = task
    return task


def get_task(task_id: str) -> Task:
    try:
        return _REGISTRY[task_id]
    except KeyError as exc:
        known = ", ".join(task_ids())
        raise KeyError(f"unknown task '{task_id}'. Known: {known}") from exc


def list_tasks() -> list[Task]:
    return [_REGISTRY[k] for k in task_ids()]


def task_ids() -> list[str]:
    return sorted(_REGISTRY)


def load_suite() -> None:
    """Import side-effect registration."""
    if _REGISTRY:
        return
    from teambench.tasks import suite as _suite  # noqa: F401


def select_tasks(ids: Sequence[str] | None = None) -> list[Task]:
    load_suite()
    if not ids:
        return list_tasks()
    return [get_task(i) for i in ids]
