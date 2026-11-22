"""Success rate, pass^k reliability, and taxonomy rollups."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from teambench.types import TrialResult


def pass_hat_k(successes: int, trials: int, k: int) -> float:
    """Unbiased estimator of p^k: C(successes, k) / C(trials, k), or 0 if k > successes.

    This is the probability that *all* k independent draws succeed, estimated
    from `trials` i.i.d. Bernoulli outcomes. It is not pass@k (at-least-one).
    """
    if k <= 0 or trials <= 0 or k > trials or successes < k:
        return 0.0
    # C(s, k) / C(n, k) = prod_{i=0..k-1} (s-i)/(n-i)
    value = 1.0
    for i in range(k):
        value *= (successes - i) / (trials - i)
    return value


def pass_at_k(successes: int, trials: int, k: int) -> float:
    """Unbiased pass@k: 1 - C(n-s, k) / C(n, k). Provided for contrast only."""
    if k <= 0 or trials <= 0 or k > trials:
        return 0.0
    failures = trials - successes
    if failures < k:
        return 1.0
    value = 1.0
    for i in range(k):
        value *= (failures - i) / (trials - i)
    return 1.0 - value


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(
    trials: Sequence[TrialResult],
    *,
    k: int | None = None,
) -> dict[str, Any]:
    """Aggregate by pattern, by task, and by pattern x task."""
    if not trials:
        return {"k": k or 0, "overall": {}, "by_pattern": {}, "by_task": {}, "by_cell": {}, "taxonomy": {}}

    max(t.trial for t in trials) + 1 if trials else 0
    # trial is 0-indexed; number of repeats is max+1, but also count unique trials per cell
    cell_counts: dict[tuple[str, str], int] = defaultdict(int)
    for trial in trials:
        cell_counts[(trial.pattern, trial.task_id)] += 1
    inferred_k = max(cell_counts.values()) if cell_counts else 0
    k_use = k or inferred_k

    def pack(rows: Sequence[TrialResult]) -> dict[str, Any]:
        n = len(rows)
        s = sum(1 for r in rows if r.success)
        return {
            "trials": n,
            "successes": s,
            "success_rate": (s / n) if n else 0.0,
            "pass_hat_k": pass_hat_k(s, n, k_use) if n else 0.0,
            "pass_at_k": pass_at_k(s, n, k_use) if n else 0.0,
            "k": k_use,
            "mean_steps": _mean([float(r.steps) for r in rows]),
            "mean_prompt_tokens": _mean([float(r.prompt_tokens) for r in rows]),
            "mean_completion_tokens": _mean([float(r.completion_tokens) for r in rows]),
            "total_tokens": sum(r.prompt_tokens + r.completion_tokens for r in rows),
            "total_cost_usd": round(sum(r.cost_usd for r in rows), 8),
            "mean_elapsed_ms": _mean([float(r.elapsed_ms) for r in rows]),
        }

    by_pattern: dict[str, list[TrialResult]] = defaultdict(list)
    by_task: dict[str, list[TrialResult]] = defaultdict(list)
    by_cell: dict[str, list[TrialResult]] = defaultdict(list)
    tags: Counter[str] = Counter()
    for row in trials:
        by_pattern[row.pattern].append(row)
        by_task[row.task_id].append(row)
        by_cell[f"{row.pattern}::{row.task_id}"].append(row)
        if not row.success:
            tags.update(row.failures or ["goal_miss"])

    return {
        "k": k_use,
        "repeats": inferred_k,
        "overall": pack(list(trials)),
        "by_pattern": {key: pack(by_pattern[key]) for key in sorted(by_pattern)},
        "by_task": {key: pack(by_task[key]) for key in sorted(by_task)},
        "by_cell": {key: pack(by_cell[key]) for key in sorted(by_cell)},
        "taxonomy": dict(tags),
    }


def finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0
