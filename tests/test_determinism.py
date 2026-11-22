from __future__ import annotations

from teambench.models.mock import ScriptedModel
from teambench.orchestrators.base import run_pattern
from teambench.runner import run_benchmark, trial_seed
from teambench.tasks.catalog import get_task, load_suite
from teambench.types import Pattern


def test_same_seed_same_outcome() -> None:
    load_suite()
    model = ScriptedModel(fallibility=0.1)
    task = get_task("T13")
    a = run_pattern(task, Pattern.SINGLE, model, trial=2, seed=99)
    b = run_pattern(task, Pattern.SINGLE, model, trial=2, seed=99)
    assert a.success == b.success
    assert a.failures == b.failures
    assert a.steps == b.steps
    assert a.submitted == b.submitted


def test_trial_seed_is_stable() -> None:
    assert trial_seed(2026, "T01", "single", 0) == trial_seed(2026, "T01", "single", 0)
    assert trial_seed(2026, "T01", "single", 0) != trial_seed(2026, "T01", "single", 1)
    assert trial_seed(2026, "T01", "single", 0) != trial_seed(2026, "T01", "debate", 0)


def test_small_benchmark_payload_shape() -> None:
    payload = run_benchmark(
        ScriptedModel(fallibility=0.0),
        task_ids=["T14"],
        patterns=["single", "debate"],
        repeats=2,
        seed=7,
        created_at="2026-11-21T00:00:00+00:00",
        notes="unit",
    )
    assert payload["manifest"]["version"]
    assert len(payload["trials"]) == 4
    singles = [t for t in payload["trials"] if t["pattern"] == "single"]
    debates = [t for t in payload["trials"] if t["pattern"] == "debate"]
    assert all(t["success"] is False for t in singles)
    assert all(t["success"] is True for t in debates)
