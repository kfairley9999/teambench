"""Run a matrix of tasks, patterns, and repeated trials."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from teambench import __version__
from teambench.models.base import ModelClient
from teambench.orchestrators.base import parse_patterns, run_pattern
from teambench.tasks.catalog import Task, select_tasks
from teambench.types import RunManifest, TrialResult


def trial_seed(base_seed: int, task_id: str, pattern: str, trial: int) -> int:
    """Stable per-cell seed so a published run is reproducible."""
    acc = base_seed
    for ch in f"{task_id}:{pattern}":
        acc = (acc * 131 + ord(ch)) % 2_147_483_647
    return (acc + 1009 * trial) % 2_147_483_647


def run_benchmark(
    model: ModelClient,
    *,
    task_ids: Sequence[str] | None = None,
    patterns: Sequence[str] | None = None,
    repeats: int = 5,
    seed: int = 2026,
    created_at: str | None = None,
    notes: str = "",
    tasks: Sequence[Task] | None = None,
) -> dict[str, Any]:
    selected = list(tasks) if tasks is not None else select_tasks(task_ids)
    chosen = parse_patterns(patterns)
    stamp = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    results: list[TrialResult] = []
    for task in selected:
        for pattern in chosen:
            for trial in range(repeats):
                cell_seed = trial_seed(seed, task.id, pattern.value, trial)
                results.append(run_pattern(task, pattern, model, trial=trial, seed=cell_seed))
    manifest = RunManifest(
        version=__version__,
        created_at=stamp,
        model=model.name,
        seed=seed,
        repeats=repeats,
        fallibility=float(getattr(model, "fallibility", 0.0)),
        patterns=[p.value for p in chosen],
        task_ids=[t.id for t in selected],
        notes=notes,
    )
    return {
        "manifest": manifest.to_dict(),
        "trials": [row.to_dict() for row in results],
    }


def trials_from_payload(payload: dict[str, Any]) -> list[TrialResult]:
    return [TrialResult.from_dict(row) for row in payload.get("trials", [])]
