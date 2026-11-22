from __future__ import annotations

from teambench.metrics import pass_at_k, pass_hat_k, summarize
from teambench.types import TrialResult


def _trial(task: str, pattern: str, trial: int, success: bool) -> TrialResult:
    return TrialResult(
        task_id=task,
        pattern=pattern,
        trial=trial,
        seed=1,
        success=success,
        failures=[] if success else ["goal_miss"],
        steps=3,
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=0.0,
        elapsed_ms=1,
        reasons=[],
    )


def test_pass_hat_k_matches_combination_ratio() -> None:
    # 4 successes out of 5, k=3 → C(4,3)/C(5,3) = 4/10 = 0.4
    assert abs(pass_hat_k(4, 5, 3) - 0.4) < 1e-12
    assert pass_hat_k(2, 5, 3) == 0.0
    assert pass_hat_k(5, 5, 5) == 1.0
    assert pass_hat_k(0, 5, 1) == 0.0


def test_pass_at_k_at_least_one() -> None:
    # 4 successes / 1 failure, k=1 → 4/5
    assert abs(pass_at_k(4, 5, 1) - 0.8) < 1e-12
    assert pass_at_k(4, 5, 2) == 1.0  # only 1 failure, cannot draw 2 failures


def test_summarize_groups_cells() -> None:
    rows = [
        _trial("T01", "single", 0, True),
        _trial("T01", "single", 1, True),
        _trial("T01", "single", 2, False),
        _trial("T01", "debate", 0, True),
        _trial("T01", "debate", 1, True),
        _trial("T01", "debate", 2, True),
    ]
    metrics = summarize(rows, k=3)
    assert metrics["overall"]["successes"] == 5
    assert metrics["by_pattern"]["debate"]["success_rate"] == 1.0
    assert abs(metrics["by_pattern"]["single"]["pass_hat_k"] - 0.0) < 1e-12  # only 2 successes
    assert metrics["taxonomy"]["goal_miss"] == 1
